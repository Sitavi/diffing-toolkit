"""HTTP surface of the crosscoder server.

Two endpoints do the work. `/measure` runs a text through both models, stacks the
two residual streams the way the crosscoder was trained on and reports the
requested latents per token. `/generate` continues a prompt, optionally steered
by a latent's decoder direction.

All GPU work goes through the `Backend` seam, so the same app serves the classic
sequential backend and any faster implementation of that protocol.
"""

import torch as th
from fastapi import FastAPI
from omegaconf import DictConfig
from pydantic import BaseModel
from tiny_dashboard.utils import apply_chat

from diffing.serving.backend import Backend, ModelName, SteeringSpec
from diffing.utils.dictionary.steering import get_crosscoder_latent

FT_SIDE = 1


class MeasureRequest(BaseModel):
    """Text to measure and which latents to report.

    `raw=True` skips chat formatting; by default the text is measured as a chat
    turn, the distribution the crosscoder statistics were computed on.
    """

    text: str
    latents: list[int]
    raw: bool = False


class GenerateRequest(BaseModel):
    """Prompt to continue, optionally steered by one latent.

    `strength` is a fraction of the latent's max activation, the unit every
    toolkit steering surface uses; `raw=True` skips chat formatting.
    """

    model: ModelName
    prompt: str
    max_new_tokens: int = 128
    latent: int | None = None
    strength: float | None = None
    raw: bool = False


def build_app(
    backend: Backend, crosscoder: th.nn.Module, cfg: DictConfig, max_acts: th.Tensor
) -> FastAPI:
    """Build the FastAPI app serving `crosscoder` through `backend`.

    Args:
        backend: Owns the models, the tokenizer and all GPU work.
        crosscoder: Trained crosscoder for `backend.layer`, already on its device.
        cfg: Composed toolkit config: identifies the run for `/status` and
            provides the chat-formatting and ignored-prefix settings.
        max_acts: Per-latent max activation [dict_size] from the latent df, the
            reference scale for steering strengths.
    """
    assert (
        crosscoder.num_layers == 2
    ), f"Expected a base/ft crosscoder, got {crosscoder.num_layers} sides"
    assert max_acts.shape == (
        crosscoder.dict_size,
    ), f"Expected max_acts [{crosscoder.dict_size}], got {tuple(max_acts.shape)}"

    enable_thinking = cfg.diffing.method.analysis.latent_steering.enable_thinking
    skip_tokens = cfg.model.ignore_first_n_tokens_per_sample_during_training

    def format_text(text: str, raw: bool) -> str:
        """Chat-format `text` the way every toolkit path does, unless `raw`."""
        if raw:
            return text
        return apply_chat(
            text, backend.tokenizer, add_bos=False, enable_thinking=enable_thinking
        )

    app = FastAPI(title="crosscoder-serve")

    @app.get("/health")
    def health() -> dict:
        """Liveness only: says nothing about whether the GPU is busy."""
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict:
        """What this process is serving."""
        return {
            "backend": type(backend).__name__,
            "model": cfg.model.name,
            "organism": cfg.organism.name,
            "layer": backend.layer,
            "dict_size": crosscoder.dict_size,
            "activation_dim": crosscoder.activation_dim,
        }

    @app.post("/measure")
    def measure(request: MeasureRequest) -> dict:
        """Per-token crosscoder activations of the requested latents.

        Activations are the code-normalized ones the analysis pipeline collects,
        so they are on the same scale as the stored max-activation statistics.
        Peaks are searched after the first `skip_tokens` positions, mirroring the
        positions those statistics exclude; a latent that never fires there is
        reported with `fired: false`.
        """
        assert len(request.latents) > 0, "latents must not be empty"
        latents = th.tensor(request.latents, dtype=th.long)
        assert (
            latents.min() >= 0 and latents.max() < crosscoder.dict_size
        ), f"Latents out of range [0, {crosscoder.dict_size})"

        token_ids = backend.tokenizer(
            format_text(request.text, request.raw), add_special_tokens=True
        )["input_ids"]
        num_tokens = len(token_ids)
        assert num_tokens > skip_tokens, (
            f"Text tokenizes to {num_tokens} tokens, "
            f"all within the {skip_tokens} ignored positions"
        )

        base_activations = backend.get_activations("base", token_ids, backend.layer)
        ft_activations = backend.get_activations("ft", token_ids, backend.layer)
        assert (
            base_activations.shape == ft_activations.shape
        ), f"Model activations disagree: {tuple(base_activations.shape)} vs {tuple(ft_activations.shape)}"
        assert (
            base_activations.shape[0] == num_tokens
        ), f"Expected {num_tokens} positions, got {base_activations.shape[0]}"

        stacked = th.stack([base_activations, ft_activations], dim=1)
        assert stacked.shape == (
            num_tokens,
            2,
            crosscoder.activation_dim,
        ), f"Expected [{num_tokens}, 2, {crosscoder.activation_dim}], got {tuple(stacked.shape)}"

        with th.no_grad():
            activations = crosscoder.get_activations(
                stacked.to(crosscoder.device, crosscoder.dtype)
            )[:, latents]
        assert activations.shape == (
            num_tokens,
            len(request.latents),
        ), f"Expected [{num_tokens}, {len(request.latents)}], got {tuple(activations.shape)}"

        peak_values, peak_positions = activations[skip_tokens:].max(dim=0)
        peak_positions = peak_positions + skip_tokens
        tokens = backend.tokenizer.convert_ids_to_tokens(token_ids)
        return {
            "layer": backend.layer,
            "raw": request.raw,
            "tokens": tokens,
            "latents": request.latents,
            "activations": activations.T.tolist(),
            "peaks": [
                {
                    "latent": latent,
                    "fired": value > 0,
                    "token_index": position if value > 0 else None,
                    "token": tokens[position] if value > 0 else None,
                    "activation": value,
                }
                for latent, position, value in zip(
                    request.latents,
                    peak_positions.tolist(),
                    peak_values.tolist(),
                )
            ],
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict:
        """Continue a prompt, optionally steered by one latent.

        The applied factor is `strength * max_act` of the latent, echoed back as
        `steering_factor`.
        """
        assert (request.latent is None) == (
            request.strength is None
        ), "latent and strength must be given together"
        factor = None
        steering = None
        if request.latent is not None:
            assert (
                0 <= request.latent < crosscoder.dict_size
            ), f"Latent {request.latent} out of range [0, {crosscoder.dict_size})"
            factor = request.strength * max_acts[request.latent].item()
            assert th.isfinite(
                th.tensor(factor)
            ), f"Latent {request.latent} has no finite max_act"
            steering = SteeringSpec(
                vector=get_crosscoder_latent(
                    request.latent, crosscoder, layer=FT_SIDE
                ).clone(),
                strength=factor,
            )
        return {
            "steering_factor": factor,
            "text": backend.generate(
                request.model,
                format_text(request.prompt, request.raw),
                steering,
                request.max_new_tokens,
            ),
        }

    return app
