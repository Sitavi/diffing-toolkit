"""Compute backends for the crosscoder server.

`Backend` is the seam between the HTTP layer and whatever holds the models: the
server only ever asks for residual-stream activations and for generated text, so
an alternative implementation (e.g. a batched vLLM engine) can be swapped in and
benchmarked against the classic path without touching `server.py`.

`ClassicBackend` is that classic path: the toolkit's nnsight models, driven one
request at a time.
"""

from threading import Lock
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch as th
from omegaconf import DictConfig
from nnterp import StandardizedTransformer
from transformers import PreTrainedTokenizerBase

ModelName = Literal["base", "ft"]


@dataclass
class SteeringSpec:
    """Residual-stream steering: `strength * vector` added at the served layer."""

    vector: th.Tensor
    strength: float

    def __post_init__(self) -> None:
        assert (
            self.vector.ndim == 1
        ), f"Steering vector must be [D], got {tuple(self.vector.shape)}"


@dataclass
class AblationSpec:
    """Directional ablation: the component along `vector` is projected out of the
    residual stream at the served layer, at every position.

    This removes the direction, not one latent's own reconstruction. The crosscoder
    encodes a STACKED [base, ft] pair, so a latent's activation at a generated
    position needs the base model's residual stream for that same prefix — which does
    not exist while the finetuned model generates its own continuation. Projecting the
    decoder direction out needs only the direction, and it is the STRONGER
    intervention: it also removes what other latents wrote along it, so behaviour that
    survives it is behaviour this direction does not carry.
    """

    vector: th.Tensor

    def __post_init__(self) -> None:
        assert (
            self.vector.ndim == 1
        ), f"Ablation vector must be [D], got {tuple(self.vector.shape)}"
        assert (
            self.vector.norm() > 0
        ), "Ablation vector is all zeros; there is no direction to remove"


def project_out(activations: th.Tensor, unit: th.Tensor) -> th.Tensor:
    """`activations` with their component along the unit vector `unit` removed.

    Args:
        activations: Residual stream [..., D].
        unit: Unit-norm direction [D], already on the activations' device.
    """
    assert (
        activations.shape[-1] == unit.shape[0]
    ), f"Expected activations [..., {unit.shape[0]}], got {tuple(activations.shape)}"
    dtype = activations.dtype
    values = activations.to(th.float32)
    coefficients = (values * unit).sum(dim=-1, keepdim=True)
    return (values - coefficients * unit).to(dtype)


@runtime_checkable
class Backend(Protocol):
    """Everything the server needs from the process that owns the GPU.

    Implementations decide how work is scheduled; the server assumes nothing
    beyond these members. `device` is where tensors handed to the backend (and
    the crosscoder joined to it) must live.
    """

    tokenizer: PreTrainedTokenizerBase
    layer: int
    device: str

    def get_activations(
        self, model: ModelName, token_ids: list[int], layer: int
    ) -> th.Tensor:
        """Residual stream of `model` after block `layer`, shaped [T, D]."""
        ...

    def generate(
        self,
        model: ModelName,
        prompt: str,
        steering: SteeringSpec | None,
        max_new_tokens: int,
        ablation: "AblationSpec | None" = None,
    ) -> str:
        """Greedily continue `prompt`, returning the continuation only.

        When `steering` is given it is applied at every position, prompt
        included — the toolkit's `all_tokens` steering mode. `ablation` is
        applied the same way, and after steering when both are given.
        """
        ...


class ClassicBackend:
    """Sequential nnsight backend: one request touches the GPU at a time.

    Both models stay resident and a single lock serializes every trace and
    generation. That is the design, not a limitation: it is the baseline a
    batched backend has to beat.

    Args:
        models: Loaded base and finetuned models, keyed as the seam names them.
        tokenizer: Tokenizer shared by both models.
        layer: Absolute layer index the crosscoder was trained on, used as the
            steering site and as the server's default measurement layer.
        disable_compile: Passed through to generation, as elsewhere in the toolkit.
    """

    def __init__(
        self,
        models: dict[ModelName, StandardizedTransformer],
        tokenizer: PreTrainedTokenizerBase,
        layer: int,
        disable_compile: bool,
    ):
        assert set(models) == {"base", "ft"}, f"Expected base and ft, got {set(models)}"
        assert layer >= 0, f"Layer must be non-negative, got {layer}"
        self.models = models
        self.tokenizer = tokenizer
        self.layer = layer
        self.disable_compile = disable_compile
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.lock = Lock()

    @classmethod
    def from_config(cls, cfg: DictConfig, layer: int) -> "ClassicBackend":
        """Build a backend from a composed toolkit config.

        Loads both models through the same calls `DiffingMethod` uses, so the
        global model cache and the config's device map, dtype and adapters apply
        unchanged.
        """
        from diffing.utils.configs import get_model_configurations
        from diffing.utils.model import load_model_from_config

        base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)
        models: dict[ModelName, StandardizedTransformer] = {
            "base": load_model_from_config(base_model_cfg),
            "ft": load_model_from_config(finetuned_model_cfg),
        }
        for model in models.values():
            model.eval()
        assert (
            models["base"].tokenizer.get_vocab() == models["ft"].tokenizer.get_vocab()
        ), "Base and finetuned tokenizers disagree; one shared tokenization is unsound"
        return cls(
            models=models,
            tokenizer=models["base"].tokenizer,
            layer=layer,
            disable_compile=base_model_cfg.disable_compile,
        )

    @th.no_grad()
    def get_activations(
        self, model: ModelName, token_ids: list[int], layer: int
    ) -> th.Tensor:
        """Residual stream of `model` after block `layer`, shaped [T, D]."""
        assert len(token_ids) > 0, "token_ids must not be empty"
        net = self.models[model]
        input_ids = th.tensor([token_ids], dtype=th.long)

        with self.lock:
            with net.trace(input_ids):
                activations = net.layers_output[layer].save()

        assert activations.shape == (
            1,
            len(token_ids),
            net.hidden_size,
        ), f"Expected [1, {len(token_ids)}, {net.hidden_size}], got {tuple(activations.shape)}"
        return activations[0].detach().cpu()

    def _unit(self, net: StandardizedTransformer, vector: th.Tensor) -> th.Tensor:
        """`vector` normalized, on the served layer's device and float32.

        The projection is computed in float32 for the same reason the causal-effect
        path does it: in bf16 the coefficient of a near-orthogonal component is noise.
        """
        if not net.dispatched:
            net.dispatch()
        param = next(net.layers[self.layer].parameters())
        unit = vector.to(device=param.device, dtype=th.float32)
        assert unit.shape == (
            net.hidden_size,
        ), f"Expected an ablation vector [{net.hidden_size}], got {tuple(unit.shape)}"
        return unit / unit.norm()

    @th.no_grad()
    def generate(
        self,
        model: ModelName,
        prompt: str,
        steering: SteeringSpec | None,
        max_new_tokens: int,
        ablation: AblationSpec | None = None,
    ) -> str:
        """Greedily continue `prompt`, returning the continuation only."""
        assert (
            max_new_tokens > 0
        ), f"max_new_tokens must be positive, got {max_new_tokens}"
        net = self.models[model]
        unit = None if ablation is None else self._unit(net, ablation.vector)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        assert input_ids.shape[0] == 1, f"Expected one prompt, got {input_ids.shape[0]}"
        prompt_len = input_ids.shape[1]

        with self.lock:
            with net.generate(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                disable_compile=self.disable_compile,
            ) as tracer:
                with tracer.invoke(input_ids):
                    if steering is not None or unit is not None:
                        for _ in tracer.all():
                            if steering is not None:
                                net.steer(
                                    self.layer,
                                    steering.vector,
                                    factor=steering.strength,
                                )
                            if unit is not None:
                                activations = net.layers_output[self.layer]
                                net.layers_output[self.layer] = project_out(
                                    activations, unit
                                )
                with tracer.invoke():
                    outputs = net.generator.output.save()

        assert outputs.shape[0] == 1, f"Expected one sequence, got {outputs.shape[0]}"
        assert (
            outputs.shape[1] >= prompt_len
        ), f"Output shorter than the prompt: {outputs.shape[1]} < {prompt_len}"
        return self.tokenizer.decode(
            outputs[0, prompt_len:].tolist(), skip_special_tokens=True
        )
