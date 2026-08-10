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

from diffing.serving.backend import Backend, ModelName, SteeringSpec

FT_SIDE = 1


class MeasureRequest(BaseModel):
    """Text to measure and which latents to report."""

    text: str
    latents: list[int]
    layer: int | None = None


class GenerateRequest(BaseModel):
    """Prompt to continue, optionally steered by one latent."""

    model: ModelName
    prompt: str
    max_new_tokens: int = 128
    latent: int | None = None
    strength: float | None = None


def steering_from_latent(
    crosscoder: th.nn.Module, latent: int, strength: float
) -> SteeringSpec:
    """Steering spec from the finetuned side of `latent`'s decoder row.

    Same vector as `diffing.utils.dictionary.steering.get_crosscoder_latent`,
    recomputed here because that module imports streamlit, which has no place in
    a served process.
    """
    assert (
        0 <= latent < crosscoder.dict_size
    ), f"Latent {latent} out of range [0, {crosscoder.dict_size})"
    return SteeringSpec(
        vector=crosscoder.decoder.weight[FT_SIDE, latent, :].detach(),
        strength=strength,
    )


def build_app(backend: Backend, crosscoder: th.nn.Module, cfg: DictConfig) -> FastAPI:
    """Build the FastAPI app serving `crosscoder` through `backend`.

    Args:
        backend: Owns the models, the tokenizer and all GPU work.
        crosscoder: Trained crosscoder for `backend.layer`, already on its device.
        cfg: Composed toolkit config, reported by `/status` to identify the run.
    """
    assert (
        crosscoder.num_layers == 2
    ), f"Expected a base/ft crosscoder, got {crosscoder.num_layers} sides"

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
        """
        assert len(request.latents) > 0, "latents must not be empty"
        latents = th.tensor(request.latents, dtype=th.long)
        assert (
            latents.min() >= 0 and latents.max() < crosscoder.dict_size
        ), f"Latents out of range [0, {crosscoder.dict_size})"

        layer = backend.layer if request.layer is None else request.layer
        token_ids = backend.tokenizer(request.text, add_special_tokens=True)[
            "input_ids"
        ]
        num_tokens = len(token_ids)

        base_activations = backend.get_activations("base", token_ids, layer)
        ft_activations = backend.get_activations("ft", token_ids, layer)
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

        activations = crosscoder.get_activations(
            stacked.to(crosscoder.device, crosscoder.dtype)
        )[:, latents]
        assert activations.shape == (
            num_tokens,
            len(request.latents),
        ), f"Expected [{num_tokens}, {len(request.latents)}], got {tuple(activations.shape)}"

        peak_values, peak_positions = activations.max(dim=0)
        tokens = backend.tokenizer.convert_ids_to_tokens(token_ids)
        return {
            "layer": layer,
            "tokens": tokens,
            "latents": request.latents,
            "activations": activations.T.tolist(),
            "peaks": [
                {
                    "latent": latent,
                    "token_index": position,
                    "token": tokens[position],
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
        """Continue a prompt, optionally steered by one latent."""
        assert (request.latent is None) == (
            request.strength is None
        ), "latent and strength must be given together"
        steering = (
            None
            if request.latent is None
            else steering_from_latent(crosscoder, request.latent, request.strength)
        )
        return {
            "text": backend.generate(
                request.model, request.prompt, steering, request.max_new_tokens
            )
        }

    return app
