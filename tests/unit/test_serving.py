"""CPU tests for the crosscoder serving stack.

Every model, tokenizer and crosscoder here is a fake chosen so that the expected
numbers can be written down by hand: activations are `token_id + dim`, latent `f`
reads off dimension `f` of the finetuned-minus-base difference, and the fake
crosscoder's decoder is an arange. No GPU, no network, no real weights.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest
import torch as th
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from diffing.serving.backend import Backend, ClassicBackend, SteeringSpec
from diffing.serving.server import build_app, steering_from_latent

VOCAB = ["the", "cake", "is", "a", "lie"]
ACTIVATION_DIM = 4
DICT_SIZE = 3
LAYER = 7
CONTINUATION_IDS = [3, 4]


class FakeTokenizer:
    """Whitespace tokenizer over a fixed vocabulary; unknown words raise."""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        token_ids = [self.vocab.index(word) for word in text.split()]
        if return_tensors == "pt":
            return {"input_ids": th.tensor([token_ids], dtype=th.long)}
        return {"input_ids": token_ids}

    def convert_ids_to_tokens(self, token_ids):
        return [self.vocab[token_id] for token_id in token_ids]

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(self.vocab[token_id] for token_id in token_ids)


class FakeCrosscoder(th.nn.Module):
    """Crosscoder-shaped stand-in with a hand-computable read-off.

    Latent `f` of token `t` is `(ft - base)[t, f]`, so its value follows directly
    from the activations the backend reports. The decoder is an arange, which
    makes each steering row identifiable.
    """

    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = 2
        self.decoder = th.nn.Module()
        self.decoder.weight = th.nn.Parameter(
            th.arange(2 * dict_size * activation_dim, dtype=th.float32).reshape(
                2, dict_size, activation_dim
            )
        )
        self.readoff = th.eye(dict_size, activation_dim)

    @property
    def device(self) -> th.device:
        return self.decoder.weight.device

    @property
    def dtype(self) -> th.dtype:
        return self.decoder.weight.dtype

    def get_activations(self, activations: th.Tensor) -> th.Tensor:
        assert activations.shape[1:] == (
            2,
            self.activation_dim,
        ), f"Expected [T, 2, {self.activation_dim}], got {tuple(activations.shape)}"
        return (activations[:, 1] - activations[:, 0]) @ self.readoff.T


class FakeBackend:
    """Backend seam stand-in that records what the server asked it for.

    Base activations are zero and finetuned activations are `token_id + dim`, so
    the crosscoder sees exactly `token_id + latent`.
    """

    def __init__(self, tokenizer: FakeTokenizer, layer: int):
        self.tokenizer = tokenizer
        self.layer = layer
        self.requested_layers: list[int] = []
        self.last_steering: SteeringSpec | None = None

    def get_activations(self, model, token_ids, layer):
        self.requested_layers.append(layer)
        dims = th.arange(ACTIVATION_DIM, dtype=th.float32)
        finetuned = th.tensor(token_ids, dtype=th.float32).unsqueeze(-1) + dims
        return th.zeros_like(finetuned) if model == "base" else finetuned

    def generate(self, model, prompt, steering, max_new_tokens):
        self.last_steering = steering
        if steering is None:
            return f"{prompt} -> {model}"
        return f"{prompt} -> {model} steered by {steering.strength}"


class SerializationProbe:
    """Records whether two calls into the fake models were ever in flight together."""

    def __init__(self):
        self.active = 0
        self.overlapped = False
        self.guard = Lock()

    def enter(self) -> None:
        with self.guard:
            self.active += 1
            self.overlapped = self.overlapped or self.active > 1

    def leave(self) -> None:
        with self.guard:
            self.active -= 1


class _Saved:
    """Stands in for an nnsight proxy whose value is already available."""

    def __init__(self, value):
        self._value = value

    def save(self):
        return self._value


class _Layers:
    def __init__(self, model: "FakeModel"):
        self._model = model

    def __getitem__(self, layer: int) -> _Saved:
        return _Saved(self._model.activations(layer))


class _Generator:
    def __init__(self, model: "FakeModel"):
        self._model = model

    @property
    def output(self) -> _Saved:
        return _Saved(self._model.generation())


class _Section:
    """Body of a trace or an invoke; the probe watches its extent."""

    def __init__(self, model: "FakeModel", input_ids: th.Tensor | None):
        self._model = model
        self._input_ids = input_ids

    def __enter__(self) -> "_Section":
        if self._input_ids is not None:
            self._model.last_input_ids = self._input_ids
        if self._model.probe is not None:
            self._model.probe.enter()
            time.sleep(self._model.delay)
        return self

    def __exit__(self, *exc_info) -> bool:
        if self._model.probe is not None:
            self._model.probe.leave()
        return False


class _Tracer:
    def __init__(self, model: "FakeModel"):
        self._model = model

    def __enter__(self) -> "_Tracer":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def invoke(self, input_ids: th.Tensor | None = None) -> _Section:
        return _Section(self._model, input_ids)

    def all(self) -> _Section:
        return _Section(self._model, None)


class FakeModel:
    """nnsight-shaped stand-in for a StandardizedTransformer.

    Residual activations are `token_id + dim`; generation appends a fixed
    continuation to whatever prompt it was invoked with; `steer` is recorded
    rather than applied.
    """

    def __init__(
        self,
        hidden_size: int,
        probe: SerializationProbe | None = None,
        delay: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.probe = probe
        self.delay = delay
        self.layers_output = _Layers(self)
        self.generator = _Generator(self)
        self.steer_calls: list[tuple[int, th.Tensor, float]] = []
        self.generate_kwargs: dict = {}
        self.last_input_ids: th.Tensor | None = None

    def activations(self, layer: int) -> th.Tensor:
        dims = th.arange(self.hidden_size, dtype=th.float32)
        return self.last_input_ids.unsqueeze(-1).float() + dims

    def generation(self) -> th.Tensor:
        return th.cat(
            [self.last_input_ids, th.tensor([CONTINUATION_IDS], dtype=th.long)], dim=1
        )

    def steer(self, layer: int, steering_vector: th.Tensor, factor: float = 1) -> None:
        self.steer_calls.append((layer, steering_vector, factor))

    def trace(self, input_ids: th.Tensor) -> _Section:
        return _Section(self, input_ids)

    def generate(self, **kwargs) -> _Tracer:
        self.generate_kwargs = kwargs
        return _Tracer(self)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer(VOCAB)


@pytest.fixture
def crosscoder() -> FakeCrosscoder:
    return FakeCrosscoder(ACTIVATION_DIM, DICT_SIZE)


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {"model": {"name": "fake_model"}, "organism": {"name": "fake_organism"}}
    )


@pytest.fixture
def backend(tokenizer) -> FakeBackend:
    return FakeBackend(tokenizer, LAYER)


@pytest.fixture
def client(backend, crosscoder, cfg) -> TestClient:
    return TestClient(build_app(backend, crosscoder, cfg))


def classic_backend(tokenizer, probe=None, delay=0.0) -> ClassicBackend:
    """A real ClassicBackend wired to fake nnsight models."""
    return ClassicBackend(
        models={
            name: FakeModel(ACTIVATION_DIM, probe=probe, delay=delay)
            for name in ("base", "ft")
        },
        tokenizer=tokenizer,
        layer=LAYER,
        disable_compile=True,
    )


def test_the_seam_is_a_structural_contract(tokenizer, backend):
    """Backends are recognised by their members, not by inheritance."""
    assert isinstance(backend, Backend)
    assert isinstance(classic_backend(tokenizer), Backend)

    class MissingGenerate:
        tokenizer = None
        layer = 0

        def get_activations(self, model, token_ids, layer):
            raise NotImplementedError

    assert not isinstance(MissingGenerate(), Backend)


def test_health_and_status_report_the_served_run(client):
    """/health is liveness only; /status identifies the run and the dictionary."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/status").json() == {
        "backend": "FakeBackend",
        "model": "fake_model",
        "organism": "fake_organism",
        "layer": LAYER,
        "dict_size": DICT_SIZE,
        "activation_dim": ACTIVATION_DIM,
    }


def test_measure_returns_hand_computed_activations(client):
    """Latent f of token t is token_id + f, one row per requested latent."""
    response = client.post(
        "/measure", json={"text": "cake is a lie", "latents": [0, 2]}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["tokens"] == ["cake", "is", "a", "lie"]
    assert body["latents"] == [0, 2]
    assert body["layer"] == LAYER
    assert body["activations"] == [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]


def test_measure_reports_the_peak_token_per_latent(client):
    """The peak is the argmax over positions, reported with its token."""
    body = client.post("/measure", json={"text": "lie cake is", "latents": [1]}).json()

    assert body["tokens"] == ["lie", "cake", "is"]
    assert body["activations"] == [[5.0, 2.0, 3.0]]
    assert body["peaks"] == [
        {"latent": 1, "token_index": 0, "token": "lie", "activation": 5.0}
    ]


def test_measure_uses_the_backend_layer_unless_overridden(client, backend):
    """The request may point the measurement at another layer."""
    client.post("/measure", json={"text": "cake", "latents": [0]})
    assert backend.requested_layers == [LAYER, LAYER]

    body = client.post(
        "/measure", json={"text": "cake", "latents": [0], "layer": 2}
    ).json()
    assert backend.requested_layers == [LAYER, LAYER, 2, 2]
    assert body["layer"] == 2


def test_measure_rejects_latents_outside_the_dictionary(client):
    """Out-of-range latents fail loudly rather than silently clamping."""
    with pytest.raises(AssertionError):
        client.post("/measure", json={"text": "cake", "latents": [DICT_SIZE]})


def test_generate_is_unsteered_without_a_latent(client, backend):
    """No latent means no steering spec reaches the backend."""
    body = client.post(
        "/generate", json={"model": "ft", "prompt": "cake is", "max_new_tokens": 4}
    ).json()

    assert backend.last_steering is None
    assert body["text"] == "cake is -> ft"


def test_generate_steers_with_the_finetuned_decoder_row(client, backend, crosscoder):
    """A latent becomes the ft-side decoder row scaled by the requested strength."""
    unsteered = client.post(
        "/generate", json={"model": "ft", "prompt": "cake is"}
    ).json()["text"]
    steered = client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake is", "latent": 2, "strength": 3.0},
    ).json()["text"]

    assert steered != unsteered
    assert backend.last_steering.strength == 3.0
    assert th.equal(backend.last_steering.vector, crosscoder.decoder.weight[1, 2, :])
    assert backend.last_steering.vector.tolist() == [20.0, 21.0, 22.0, 23.0]


def test_generate_requires_latent_and_strength_together(client):
    """A latent without a strength is a malformed request, not a default."""
    with pytest.raises(AssertionError):
        client.post("/generate", json={"model": "ft", "prompt": "cake", "latent": 1})


def test_steering_from_latent_rejects_unknown_latents(crosscoder):
    """The decoder row must exist."""
    with pytest.raises(AssertionError):
        steering_from_latent(crosscoder, DICT_SIZE, 1.0)


def test_steering_spec_requires_a_flat_vector():
    """A steering direction is [D]; anything else is a bug upstream."""
    with pytest.raises(AssertionError):
        SteeringSpec(vector=th.zeros(2, ACTIVATION_DIM), strength=1.0)


def test_classic_backend_returns_one_row_per_token(tokenizer):
    """get_activations drops the batch dimension and keeps [T, D]."""
    backend = classic_backend(tokenizer)
    activations = backend.get_activations("base", [1, 2, 3], LAYER)

    assert activations.shape == (3, ACTIVATION_DIM)
    assert activations[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert activations[2].tolist() == [3.0, 4.0, 5.0, 6.0]


def test_classic_backend_generate_returns_only_the_continuation(tokenizer):
    """The prompt is sliced off before decoding."""
    backend = classic_backend(tokenizer)
    text = backend.generate("ft", "cake is", steering=None, max_new_tokens=2)

    assert text == "a lie"
    assert backend.models["ft"].steer_calls == []
    assert backend.models["ft"].generate_kwargs["max_new_tokens"] == 2
    assert backend.models["ft"].generate_kwargs["do_sample"] is False


def test_classic_backend_steers_at_the_served_layer(tokenizer):
    """The steering spec reaches nnsight's steer at the backend's layer."""
    backend = classic_backend(tokenizer)
    vector = th.arange(ACTIVATION_DIM, dtype=th.float32)
    backend.generate(
        "ft",
        "cake is",
        steering=SteeringSpec(vector=vector, strength=2.5),
        max_new_tokens=2,
    )

    assert len(backend.models["ft"].steer_calls) == 1
    layer, steered_vector, factor = backend.models["ft"].steer_calls[0]
    assert layer == LAYER
    assert factor == 2.5
    assert th.equal(steered_vector, vector)


def test_the_lock_serializes_concurrent_gpu_work(tokenizer):
    """Two requests in flight at once queue instead of interleaving."""
    probe = SerializationProbe()
    delay = 0.05
    backend = classic_backend(tokenizer, probe=probe, delay=delay)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(backend.get_activations, model, [1, 2], LAYER)
            for model in ("base", "ft")
        ]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start

    assert not probe.overlapped
    assert elapsed >= 2 * delay
    assert all(result.shape == (2, ACTIVATION_DIM) for result in results)


def test_the_probe_sees_overlap_when_the_lock_is_bypassed(tokenizer):
    """Control: the same two calls do overlap without the backend's lock."""
    probe = SerializationProbe()
    backend = classic_backend(tokenizer, probe=probe, delay=0.05)

    def trace(model: str) -> th.Tensor:
        net = backend.models[model]
        with net.trace(th.tensor([[1, 2]], dtype=th.long)):
            return net.layers_output[LAYER].save()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trace, model) for model in ("base", "ft")]
        [future.result() for future in futures]

    assert probe.overlapped
