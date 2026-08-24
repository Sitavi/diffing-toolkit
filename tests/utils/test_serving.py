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

from diffing.serving.backend import (AblationSpec, Backend, ClassicBackend,
                                     SteeringSpec, project_out)
from contextlib import contextmanager

from diffing.serving.queue import QueueFull
from diffing.serving.server import FT_SIDE, build_app

VOCAB = ["the", "cake", "is", "a", "lie"]
ACTIVATION_DIM = 4
DICT_SIZE = 3
LAYER = 7
CONTINUATION_IDS = [3, 4]
SKIP_TOKENS = 1
MAX_ACTS = [10.0, 20.0, 30.0]


class FakeTokenizer:
    """Whitespace tokenizer over a fixed vocabulary; unknown words raise.

    Its chat template is the identity on the message content, so chat-formatted
    text tokenizes exactly like the input while `chat_calls` records that the
    formatting path was taken.
    """

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.eos_token_id = 0
        self.bos_token = None
        self.chat_calls: list[list[dict]] = []

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        token_ids = [self.vocab.index(word) for word in text.split()]
        if return_tensors == "pt":
            return {"input_ids": th.tensor([token_ids], dtype=th.long)}
        return {"input_ids": token_ids}

    def apply_chat_template(
        self, chat, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        self.chat_calls.append(chat)
        return " ".join(message["content"] for message in chat)

    def __len__(self):
        return len(self.vocab)

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
        self.last_select_features = None

    @property
    def device(self) -> th.device:
        return self.decoder.weight.device

    @property
    def dtype(self) -> th.dtype:
        return self.decoder.weight.dtype

    def get_activations(
        self, activations: th.Tensor, select_features=None
    ) -> th.Tensor:
        assert activations.shape[1:] == (
            2,
            self.activation_dim,
        ), f"Expected [T, 2, {self.activation_dim}], got {tuple(activations.shape)}"
        self.last_select_features = select_features
        full = (activations[:, 1] - activations[:, 0]) @ self.readoff.T
        return full if select_features is None else full[:, select_features]


class FakeBackend:
    """Backend seam stand-in that records what the server asked it for.

    Base activations are zero and finetuned activations are `token_id + dim`, so
    the crosscoder sees exactly `token_id + latent`.
    """

    def __init__(self, tokenizer: FakeTokenizer, layer: int):
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = "cpu"
        self.requested_layers: list[int] = []
        self.last_steering: SteeringSpec | None = None
        self.last_ablation: AblationSpec | None = None

    def get_activations(self, model, token_ids, layer):
        self.requested_layers.append(layer)
        dims = th.arange(ACTIVATION_DIM, dtype=th.float32)
        finetuned = th.tensor(token_ids, dtype=th.float32).unsqueeze(-1) + dims
        return th.zeros_like(finetuned) if model == "base" else finetuned

    def generate(self, model, prompt, steering, max_new_tokens, ablation=None):
        self.last_steering = steering
        self.last_ablation = ablation
        suffix = "" if ablation is None else " ablated"
        if steering is None:
            return f"{prompt} -> {model}{suffix}"
        return f"{prompt} -> {model} steered by {steering.strength}{suffix}"


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


class _Saved(th.Tensor):
    """Stands in for an nnsight proxy whose value is already available.

    A tensor subclass rather than a wrapper: the server both SAVES a layer's
    output (`get_activations`) and COMPUTES on it (an ablation projects it), so
    the stand-in has to answer `.save()` and behave as the tensor it stands for.
    """

    @staticmethod
    def wrap(value: th.Tensor) -> "_Saved":
        return value.as_subclass(_Saved)

    def save(self) -> "_Saved":
        return self


class _Layers:
    """Reads report the fake residual stream; writes are recorded, as an
    intervention on a real model would replace the layer's output."""

    def __init__(self, model: "FakeModel"):
        self._model = model

    def __getitem__(self, layer: int) -> _Saved:
        return _Saved.wrap(self._model.activations(layer))

    def __setitem__(self, layer: int, value: th.Tensor) -> None:
        self._model.layer_writes.append((layer, value))


class _Generator:
    def __init__(self, model: "FakeModel"):
        self._model = model

    @property
    def output(self) -> _Saved:
        return _Saved.wrap(self._model.generation())


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

    def all(self):
        return iter([None])


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
        self.dispatched = True
        self.layers = [th.nn.Linear(hidden_size, hidden_size) for _ in range(LAYER + 1)]
        self.steer_calls: list[tuple[int, th.Tensor, float]] = []
        self.layer_writes: list[tuple[int, th.Tensor]] = []
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
        {
            "model": {
                "name": "fake_model",
                "ignore_first_n_tokens_per_sample_during_training": SKIP_TOKENS,
            },
            "organism": {"name": "fake_organism"},
            "diffing": {
                "method": {
                    "analysis": {"latent_steering": {"enable_thinking": False}}
                }
            },
        }
    )


@pytest.fixture
def backend(tokenizer) -> FakeBackend:
    return FakeBackend(tokenizer, LAYER)


@pytest.fixture
def client(backend, crosscoder, cfg) -> TestClient:
    return TestClient(
        build_app(backend, crosscoder, cfg, th.tensor(MAX_ACTS, dtype=th.float32))
    )


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
    """/health is liveness only; /status identifies the run, the dictionary and how busy the
    one GPU is — a client sharing the server needs the queue depth to read a slow answer
    correctly."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/status").json() == {
        "backend": "FakeBackend",
        "model": "fake_model",
        "organism": "fake_organism",
        "layer": LAYER,
        "dict_size": DICT_SIZE,
        "activation_dim": ACTIVATION_DIM,
        "queue": {"waiting": 0, "capacity": 32, "running": False,
                  "served": 0, "rejected": 0, "mean_wait_s": 0.0},
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
    """The peak is the argmax over the non-ignored positions.

    Position 0 holds the run's maximum (5.0 on "lie") but falls inside the
    ignored prefix, so the peak is the best of the remaining positions.
    """
    body = client.post("/measure", json={"text": "lie cake is", "latents": [1]}).json()

    assert body["tokens"] == ["lie", "cake", "is"]
    assert body["activations"] == [[5.0, 2.0, 3.0]]
    assert body["peaks"] == [
        {"latent": 1, "fired": True, "token_index": 2, "token": "is", "activation": 3.0}
    ]


def test_measure_reports_silent_latents_as_not_fired(client):
    """A latent at zero everywhere gets fired=false, not a fake peak at index 0."""
    body = client.post("/measure", json={"text": "the the", "latents": [0, 1]}).json()

    assert body["activations"] == [[0.0, 0.0], [1.0, 1.0]]
    assert body["peaks"][0] == {
        "latent": 0,
        "fired": False,
        "token_index": None,
        "token": None,
        "activation": 0.0,
    }
    assert body["peaks"][1]["fired"] is True


def test_measure_computes_only_the_requested_latents(client, crosscoder):
    """The crosscoder is asked for the requested latents, not the full code."""
    client.post("/measure", json={"text": "cake is", "latents": [0, 2]})
    assert crosscoder.last_select_features == [0, 2]


def test_measure_always_uses_the_served_layer(client, backend):
    """The measurement layer is the crosscoder's; a request cannot move it."""
    body = client.post(
        "/measure", json={"text": "cake is", "latents": [0], "layer": 2}
    ).json()
    assert backend.requested_layers == [LAYER, LAYER]
    assert body["layer"] == LAYER


def test_measure_chat_formats_unless_raw(client, tokenizer):
    """Text goes through the chat template by default; raw=True bypasses it."""
    client.post("/measure", json={"text": "cake is", "latents": [0]})
    assert len(tokenizer.chat_calls) == 1
    assert tokenizer.chat_calls[0] == [{"role": "user", "content": "cake is"}]

    body = client.post(
        "/measure", json={"text": "cake is", "latents": [0], "raw": True}
    ).json()
    assert len(tokenizer.chat_calls) == 1
    assert body["raw"] is True


def test_measure_accepts_exact_token_ids(client, backend):
    """token_ids are measured verbatim: no chat formatting, no tokenization."""
    body = client.post(
        "/measure", json={"token_ids": [4, 1, 2], "latents": [1]}
    ).json()

    assert body["raw"] is True
    assert body["tokens"] == ["lie", "cake", "is"]
    assert body["surfaces"] == ["lie", "cake", "is"]
    assert body["activations"] == [[5.0, 2.0, 3.0]]
    assert backend.tokenizer.chat_calls == []


def test_measure_requires_exactly_one_input_form(client):
    """text and token_ids are exclusive, and one of them is required."""
    both = client.post(
        "/measure", json={"text": "cake is", "token_ids": [1, 2], "latents": [0]}
    )
    neither = client.post("/measure", json={"latents": [0]})
    assert both.status_code == 400
    assert neither.status_code == 400
    assert "exactly one" in both.json()["detail"].lower()


def test_measure_rejects_token_ids_outside_the_vocabulary(client):
    """An out-of-vocabulary id would crash in the embedding lookup instead."""
    response = client.post("/measure", json={"token_ids": [1, 99], "latents": [0]})
    assert response.status_code == 400
    assert "vocabulary" in response.json()["detail"]


def test_measure_rejects_latents_outside_the_dictionary(client):
    """Out-of-range latents are a client error with a message, not a 500."""
    response = client.post(
        "/measure", json={"text": "cake is", "latents": [DICT_SIZE]}
    )
    assert response.status_code == 400
    assert "out of range" in response.json()["detail"].lower()


def test_measure_rejects_empty_text_and_empty_latents(client):
    """Pydantic rejects vacuous requests before any GPU work."""
    assert (
        client.post("/measure", json={"text": "", "latents": [0]}).status_code == 422
    )
    assert (
        client.post("/measure", json={"text": "cake", "latents": []}).status_code
        == 422
    )


def test_measure_rejects_text_shorter_than_the_ignored_prefix(client):
    """A text entirely inside the ignored prefix has no measurable position."""
    response = client.post(
        "/measure", json={"text": "cake", "latents": [0], "raw": True}
    )
    assert response.status_code == 400
    assert "ignored positions" in response.json()["detail"]


def test_generate_is_unsteered_without_a_latent(client, backend):
    """No latent means no steering spec reaches the backend."""
    body = client.post(
        "/generate", json={"model": "ft", "prompt": "cake is", "max_new_tokens": 4}
    ).json()

    assert backend.last_steering is None
    assert body["text"] == "cake is -> ft"
    assert body["steering_factor"] is None


def test_generate_steers_with_the_finetuned_decoder_row(client, backend, crosscoder):
    """A latent becomes the ft-side decoder row at strength * its max_act."""
    unsteered = client.post(
        "/generate", json={"model": "ft", "prompt": "cake is"}
    ).json()["text"]
    body = client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake is", "latent": 2, "strength": 3.0},
    ).json()

    assert body["text"] != unsteered
    assert body["steering_factor"] == 3.0 * MAX_ACTS[2]
    assert backend.last_steering.strength == 3.0 * MAX_ACTS[2]
    assert th.equal(backend.last_steering.vector, crosscoder.decoder.weight[1, 2, :])
    assert backend.last_steering.vector.tolist() == [20.0, 21.0, 22.0, 23.0]


def test_generate_chat_formats_unless_raw(client, backend, tokenizer):
    """The prompt goes through the chat template by default; raw=True bypasses it."""
    client.post("/generate", json={"model": "base", "prompt": "cake is"})
    assert len(tokenizer.chat_calls) == 1

    client.post("/generate", json={"model": "base", "prompt": "cake is", "raw": True})
    assert len(tokenizer.chat_calls) == 1


def test_generate_requires_exactly_one_strength_form_with_a_latent(client):
    """A latent needs strength XOR factor; either without a latent is malformed."""
    for bad in (
        {"latent": 1},
        {"latent": 1, "strength": 1.0, "factor": 2.0},
        {"strength": 1.0},
        {"factor": 2.0},
    ):
        response = client.post(
            "/generate", json={"model": "ft", "prompt": "cake", **bad}
        )
        assert response.status_code == 400


def test_generate_steers_with_an_absolute_factor(client, backend):
    """factor bypasses the max_act conversion and is applied verbatim."""
    body = client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake is", "latent": 2, "factor": 7.5},
    ).json()

    assert body["steering_factor"] == 7.5
    assert backend.last_steering.strength == 7.5


def test_generate_echoes_the_formatted_prompt(client):
    """formatted_prompt/prompt_tokens let a client measure prompt+answer heat."""
    body = client.post(
        "/generate", json={"model": "ft", "prompt": "cake is a lie"}
    ).json()

    assert body["formatted_prompt"] == "cake is a lie"
    assert body["prompt_tokens"] == 4


def test_generate_bounds_max_new_tokens(client):
    """Zero, negative and unbounded generation lengths are client errors."""
    for bad in (0, -1, 1_000_000):
        response = client.post(
            "/generate",
            json={"model": "ft", "prompt": "cake", "max_new_tokens": bad},
        )
        assert response.status_code == 422


def test_generate_rejects_latents_outside_the_dictionary(client):
    """The decoder row must exist; negatives die in pydantic, overshoots in 400."""
    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake", "latent": DICT_SIZE, "strength": 1.0},
    )
    assert response.status_code == 400
    assert "out of range" in response.json()["detail"].lower()

    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake", "latent": -1, "strength": 1.0},
    )
    assert response.status_code == 422


def test_the_steering_vector_does_not_alias_the_decoder(client, backend, crosscoder):
    """The spec crossing the seam owns its memory, not a view of the weights."""
    client.post(
        "/generate",
        json={"model": "ft", "prompt": "cake", "latent": 1, "strength": 1.0},
    )
    vector = backend.last_steering.vector
    assert th.equal(vector, crosscoder.decoder.weight[1, 1, :])
    assert (
        vector.data_ptr() != crosscoder.decoder.weight[1, 1, :].data_ptr()
    )


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
    backend = classic_backend(tokenizer, probe=probe, delay=0.2)

    def trace(model: str) -> th.Tensor:
        net = backend.models[model]
        with net.trace(th.tensor([[1, 2]], dtype=th.long)):
            return net.layers_output[LAYER].save()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trace, model) for model in ("base", "ft")]
        [future.result() for future in futures]

    assert probe.overlapped


class _VocabTokenizer:
    def __init__(self, vocab: dict):
        self._vocab = vocab

    def get_vocab(self) -> dict:
        return self._vocab


class _LoadedModel:
    def __init__(self, vocab: dict):
        self.tokenizer = _VocabTokenizer(vocab)
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


def _patch_model_loading(monkeypatch, base_vocab: dict, ft_vocab: dict):
    """Route from_config's model loading to in-memory fakes."""
    import diffing.utils.configs as configs_module
    import diffing.utils.model as model_module

    model_cfg = OmegaConf.create({"disable_compile": True})
    loaded = [_LoadedModel(base_vocab), _LoadedModel(ft_vocab)]
    monkeypatch.setattr(
        configs_module, "get_model_configurations", lambda cfg: (model_cfg, model_cfg)
    )
    monkeypatch.setattr(
        model_module, "load_model_from_config", lambda cfg: loaded.pop(0)
    )
    return loaded


def test_from_config_builds_an_evaled_backend(monkeypatch):
    """Both models are loaded, evaled and share the base tokenizer."""
    _patch_model_loading(monkeypatch, {"a": 0}, {"a": 0})
    backend = ClassicBackend.from_config(OmegaConf.create({}), layer=3)

    assert backend.layer == 3
    assert backend.disable_compile is True
    assert all(model.eval_called for model in backend.models.values())
    assert backend.tokenizer is backend.models["base"].tokenizer


def test_from_config_rejects_disagreeing_tokenizers(monkeypatch):
    """One shared tokenization is unsound if the vocabularies differ."""
    _patch_model_loading(monkeypatch, {"a": 0}, {"a": 0, "b": 1})
    with pytest.raises(AssertionError):
        ClassicBackend.from_config(OmegaConf.create({}), layer=3)


def test_cli_accepts_overrides_interleaved_with_flags():
    """Hydra overrides parse wherever they sit relative to --host/--port."""
    from diffing.cli.crosscoder_serve import build_parser, parse_args

    args = parse_args(
        build_parser(), ["model=x", "--port", "8000", "organism=y", "a.b=1"]
    )
    assert args.overrides == ["model=x", "organism=y", "a.b=1"]
    assert args.port == 8000


def test_cli_rejects_non_override_unknowns():
    """A stray flag is an error, not a silently dropped override."""
    from diffing.cli.crosscoder_serve import build_parser, parse_args

    with pytest.raises(SystemExit):
        parse_args(build_parser(), ["model=x", "--bogus"])


def test_cli_config_dir_repeats_and_guards_searchpath():
    """--config-dir collects directories; a manual hydra.searchpath override is an error
    in that mode (the CLI owns the search path there)."""
    from diffing.cli.crosscoder_serve import build_parser, parse_args

    args = parse_args(
        build_parser(),
        ["--config-dir", "/a", "--config-dir", "/b", "--config-name", "my_run"],
    )
    assert args.config_dirs == ["/a", "/b"]
    assert args.config_name == "my_run"
    with pytest.raises(SystemExit):
        parse_args(
            build_parser(),
            ["--config-dir", "/a", "hydra.searchpath=[file:///x]"],
        )


def test_compose_config_reads_a_generated_run_file(tmp_path):
    """A generated run config (defaults extending `config` with group overrides) composes
    from its own directory, with the package configs/ resolving the base groups."""
    from diffing.cli.crosscoder_serve import compose_config

    (tmp_path / "my_run.yaml").write_text(
        "defaults:\n"
        "  - config\n"
        "  - override model: qwen3_1_7B\n"
        "  - override organism: cake_bake\n"
        "  - override diffing/method: crosscoder\n"
        "  - _self_\n"
        "\n"
        "preprocessing:\n"
        "  layers: [0.5]\n"
        "\n"
        "diffing:\n"
        "  method:\n"
        "    training:\n"
        "      expansion_factor: 16\n"
        "      k: 64\n"
        "    streaming:\n"
        "      enabled: true\n"
    )
    cfg = compose_config([str(tmp_path)], "my_run", [])
    assert cfg.model.name == "qwen3_1_7B"
    assert cfg.organism.name == "cake_bake"
    assert cfg.diffing.method.name == "crosscoder"
    assert list(cfg.preprocessing.layers) == [0.5]
    assert cfg.diffing.method.training.expansion_factor == 16
    assert cfg.diffing.method.training.k == 64
    assert cfg.diffing.method.streaming.enabled is True


def test_project_out_removes_the_direction_and_nothing_else():
    """The projection is the whole intervention, so it is worth pinning directly:
    the component along the direction goes to zero and the orthogonal rest is kept."""
    unit = th.tensor([0.0, 1.0, 0.0, 0.0])
    activations = th.tensor([[[1.0, 5.0, 2.0, 3.0], [0.0, -4.0, 1.0, 0.0]]])
    ablated = project_out(activations, unit)

    assert th.allclose((ablated * unit).sum(dim=-1), th.zeros(1, 2), atol=1e-6)
    kept = th.tensor([[[1.0, 0.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0]]])
    assert th.allclose(ablated, kept, atol=1e-6)
    assert ablated.dtype == activations.dtype


def test_project_out_keeps_the_input_dtype():
    """The residual stream is bf16 on a real model; the coefficient is computed in
    float32 but what goes back into the model must be what came out of it."""
    unit = th.tensor([1.0, 0.0, 0.0, 0.0])
    activations = th.tensor([[2.0, 7.0, 1.0, 0.0]], dtype=th.bfloat16)
    assert project_out(activations, unit).dtype == th.bfloat16


def test_generate_is_unablated_without_the_field(client, backend):
    """Ablation is opt-in: no `ablate`, no spec reaches the backend."""
    response = client.post(
        "/generate", json={"model": "ft", "prompt": "the cake", "max_new_tokens": 4}
    )
    assert response.status_code == 200
    assert backend.last_ablation is None
    assert response.json()["ablated_latent"] is None


def test_generate_ablates_the_requested_latent(client, backend, crosscoder):
    """The spec carries that latent's FINETUNED-side decoder row, the same side
    steering uses — the direction the feature writes into the served stream."""
    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "the cake", "ablate": 2, "max_new_tokens": 4},
    )
    assert response.status_code == 200
    assert response.json()["ablated_latent"] == 2
    assert th.equal(backend.last_ablation.vector, crosscoder.decoder.weight[FT_SIDE, 2])


def test_ablation_and_steering_compose(client, backend):
    """Different latents may be steered and ablated in one pass: the question
    'what does A cause once B is gone' needs both at once."""
    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "the cake", "latent": 1, "strength": 0.5,
              "ablate": 2, "max_new_tokens": 4},
    )
    assert response.status_code == 200
    assert backend.last_steering.strength == pytest.approx(0.5 * MAX_ACTS[1])
    assert backend.last_ablation is not None


def test_steering_and_ablating_one_latent_is_refused(client):
    """Adding a direction and removing it in the same pass is not a coherent request."""
    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "the cake", "latent": 2, "strength": 0.5,
              "ablate": 2, "max_new_tokens": 4},
    )
    assert response.status_code == 400
    assert "contradict" in response.json()["detail"]


def test_ablating_an_unknown_latent_is_refused(client):
    response = client.post(
        "/generate",
        json={"model": "ft", "prompt": "the cake", "ablate": DICT_SIZE,
              "max_new_tokens": 4},
    )
    assert response.status_code == 400
    assert "out of range" in response.json()["detail"]


def test_classic_backend_writes_the_ablated_stream_back(tokenizer):
    """End of the seam: the real backend must actually replace the layer's output,
    with the direction removed, at the served layer."""
    backend = classic_backend(tokenizer)
    direction = th.zeros(ACTIVATION_DIM)
    direction[1] = 3.0                      # unnormalized on purpose: the backend normalizes
    backend.generate("ft", "the cake", None, 4, ablation=AblationSpec(vector=direction))

    writes = backend.models["ft"].layer_writes
    assert [layer for layer, _ in writes] == [LAYER]
    written = writes[0][1]
    unit = direction / direction.norm()
    assert th.allclose((written.float() * unit).sum(dim=-1),
                       th.zeros(written.shape[:-1]), atol=1e-5)
    assert not backend.models["ft"].steer_calls    # ablation is not steering


# ---------------------------------------------------------------- GPU queue
def test_queue_serves_in_arrival_order():
    """FIFO is the point: a plain lock can skip a waiter repeatedly, so a long battery could
    starve the person clicking in the dashboard."""
    import threading

    from diffing.serving.queue import GpuQueue

    q = GpuQueue(capacity=8)
    order, entered = [], []
    release = threading.Event()

    with q.slot():                       # hold the GPU so everyone else must queue
        for i in range(5):
            ev = threading.Event()
            entered.append(ev)

            def worker(i=i, ev=ev):
                ev.set()
                with q.slot():
                    order.append(i)
                    release.wait(2)

            threading.Thread(target=worker, daemon=True).start()
            ev.wait(2)
            time.sleep(0.02)             # let it reach the queue, in this order
    release.set()
    for _ in range(50):
        if len(order) == 5:
            break
        time.sleep(0.02)
    assert order == [0, 1, 2, 3, 4]


def test_queue_refuses_before_it_works_never_after():
    """A full queue answers immediately. The refusal must happen before the request has cost
    anything, so a client can retry it safely."""
    import threading

    from diffing.serving.queue import GpuQueue, QueueFull

    q = GpuQueue(capacity=1)
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with q.slot():
            holding.set()
            release.wait(2)

    threading.Thread(target=hold, daemon=True).start()
    holding.wait(2)
    waiter = threading.Thread(target=lambda: q.slot().__enter__(), daemon=True)
    waiter.start()
    time.sleep(0.05)                     # one running, one waiting = at capacity
    with pytest.raises(QueueFull) as err:
        with q.slot():
            raise AssertionError("a third request was admitted past the capacity")
    assert "retry" in str(err.value)
    assert q.snapshot()["rejected"] == 1
    release.set()


def test_an_idle_queue_admits_the_first_request_at_any_capacity():
    """capacity counts WAITERS. capacity=0 means "serve one, refuse the rest" — not "refuse
    everything", which is what counting the running request would have given."""
    from diffing.serving.queue import GpuQueue

    q = GpuQueue(capacity=0)
    with q.slot():
        pass
    with q.slot():
        pass
    assert q.snapshot() == {"waiting": 0, "capacity": 0, "running": False,
                            "served": 2, "rejected": 0, "mean_wait_s": 0.0}


def test_a_failed_request_still_releases_the_gpu():
    """The one bug that would take the server down for good: an exception inside a slot that
    never advances the queue leaves every later request waiting forever."""
    from diffing.serving.queue import GpuQueue

    q = GpuQueue(capacity=4)
    with pytest.raises(ValueError):
        with q.slot():
            raise ValueError("generation blew up")
    with q.slot():
        pass
    assert q.snapshot()["served"] == 2


def test_status_reports_the_queue(client):
    body = client.get("/status").json()
    assert body["queue"]["capacity"] > 0
    assert body["queue"]["running"] is False
    assert body["queue"]["waiting"] == 0


def test_a_full_queue_is_a_503_the_client_can_act_on(backend, crosscoder, cfg):
    """Not a hang and not a 500: the work never started, the answer says so, and Retry-After
    tells a battery how long to sit out."""
    from fastapi.testclient import TestClient

    from diffing.serving.queue import GpuQueue
    from diffing.serving.server import build_app

    class _Closed(GpuQueue):
        @contextmanager
        def slot(self):
            self._rejected += 1
            raise QueueFull(99, 0)
            yield                                  # pragma: no cover

    client = TestClient(build_app(backend, crosscoder, cfg,
                                  th.tensor(MAX_ACTS, dtype=th.float32), queue=_Closed()))
    r = client.post("/generate", json={"model": "ft", "prompt": "cake is"})
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "5"
    assert "waiting for the GPU" in r.json()["detail"]
    r = client.post("/measure", json={"text": "the cake is a lie", "latents": [0]})
    assert r.status_code == 503
