"""HTTP surface of the crosscoder server.

Two endpoints do the work. `/measure` runs a text through both models, stacks the
two residual streams the way the crosscoder was trained on and reports the
requested latents per token. `/generate` continues a prompt, optionally steered
by a latent's decoder direction and/or with a direction projected out of the
residual stream (ablation).

All GPU work goes through the `Backend` seam, so the same app serves the classic
sequential backend and any faster implementation of that protocol.
"""

from contextlib import contextmanager

import torch as th
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from tiny_dashboard.utils import apply_chat

from diffing.serving.backend import AblationSpec, Backend, ModelName, SteeringSpec
from diffing.serving.queue import GpuQueue, QueueFull
from diffing.utils.dictionary.steering import get_crosscoder_latent

FT_SIDE = 1


class MeasureRequest(BaseModel):
    """Text (or exact token ids) to measure and which latents to report.

    `raw=True` skips chat formatting; by default text is measured as a chat
    turn, the distribution the crosscoder statistics were computed on.
    `token_ids` measures exactly those ids — no formatting, no tokenization —
    which is how stored examples are re-measured without decode/re-encode drift.
    """

    text: str | None = Field(default=None, min_length=1)
    token_ids: list[int] | None = Field(default=None, min_length=1)
    latents: list[int] = Field(min_length=1)
    raw: bool = False


class GenerateRequest(BaseModel):
    """Prompt to continue, optionally steered and/or ablated by latents.

    `strength` is a fraction of the latent's max activation, the unit every
    toolkit steering surface uses; `factor` is the absolute residual-stream
    multiplier instead, for callers that computed their own reference. Give
    exactly one of them with `latent`. `raw=True` skips chat formatting.

    `ablate` names a latent whose decoder DIRECTION is projected out of the
    residual stream at every position — the causal counterpart of steering:
    steering asks what the direction can cause, ablation asks what stops
    happening without it. See `AblationSpec` for why this removes the direction
    rather than that latent's own reconstruction.
    """

    model: ModelName
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, gt=0, le=2048)
    latent: int | None = Field(default=None, ge=0)
    strength: float | None = None
    factor: float | None = None
    ablate: int | None = Field(default=None, ge=0)
    raw: bool = False


def build_app(
    backend: Backend,
    crosscoder: th.nn.Module,
    cfg: DictConfig,
    max_acts: th.Tensor,
    queue: GpuQueue | None = None,
) -> FastAPI:
    """Build the FastAPI app serving `crosscoder` through `backend`.

    Args:
        backend: Owns the models, the tokenizer and all GPU work.
        crosscoder: Trained crosscoder for `backend.layer`, already on its device.
        cfg: Composed toolkit config: identifies the run for `/status` and
            provides the chat-formatting and ignored-prefix settings.
        max_acts: Per-latent max activation [dict_size] from the latent df, the
            reference scale for steering strengths.
        queue: Admission control for the GPU. Every request that touches the models goes
            through it, so a sharded battery, the dashboard and an agent can share one
            server without stampeding it. Default: a queue of 32 waiters.
    """
    queue = GpuQueue() if queue is None else queue
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

    @contextmanager
    def _gpu(what: str):
        """Hold the GPU for one section, or refuse now if too many are already waiting.

        A 503 with Retry-After is the honest answer to a full queue: the work was never
        started, so retrying is safe, and a client that is told to wait can decide how long
        it is willing to. Silently queueing a request behind forty others would look like a
        server that has hung.
        """
        try:
            with queue.slot():
                yield
        except QueueFull as full:
            raise HTTPException(
                status_code=503,
                detail=f"{what}: {full}",
                headers={"Retry-After": "5"},
            ) from None

    def _generate(formatted, steering, ablation, request) -> str:
        with _gpu("generate"):
            return backend.generate(
                request.model,
                formatted,
                steering,
                request.max_new_tokens,
                ablation=ablation,
            )

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
            # how busy the one GPU is: a client that knows it is ninth in line can wait
            # deliberately instead of reading a slow answer as a broken server
            "queue": queue.snapshot(),
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
        if not all(0 <= latent < crosscoder.dict_size for latent in request.latents):
            raise HTTPException(
                status_code=400,
                detail=f"Latents out of range [0, {crosscoder.dict_size})",
            )
        if (request.text is None) == (request.token_ids is None):
            raise HTTPException(
                status_code=400,
                detail="Exactly one of text and token_ids must be given",
            )

        raw = request.raw or request.token_ids is not None
        if request.token_ids is not None:
            if max(request.token_ids) >= len(backend.tokenizer):
                raise HTTPException(
                    status_code=400,
                    detail=f"Token ids outside the vocabulary "
                    f"[0, {len(backend.tokenizer)})",
                )
            token_ids = request.token_ids
        else:
            token_ids = backend.tokenizer(
                format_text(request.text, request.raw), add_special_tokens=True
            )["input_ids"]
        num_tokens = len(token_ids)
        if num_tokens <= skip_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Text tokenizes to {num_tokens} tokens, "
                f"all within the {skip_tokens} ignored positions",
            )

        # ONE slot for the whole measurement: two forwards and the crosscoder encode are one
        # request's worth of GPU. Taking a second slot half way through would let a request
        # be refused after its work had already been done, which is the one thing a queue
        # must not do.
        with _gpu("measure"):
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
                    stacked.to(crosscoder.device, crosscoder.dtype),
                    select_features=request.latents,
                )
        assert activations.shape == (
            num_tokens,
            len(request.latents),
        ), f"Expected [{num_tokens}, {len(request.latents)}], got {tuple(activations.shape)}"

        peak_values, peak_positions = activations[skip_tokens:].max(dim=0)
        peak_positions = peak_positions + skip_tokens
        tokens = [backend.tokenizer.decode([token_id]) for token_id in token_ids]
        return {
            "layer": backend.layer,
            "raw": raw,
            "tokens": tokens,
            "surfaces": backend.tokenizer.convert_ids_to_tokens(token_ids),
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
        """Continue a prompt, optionally steered and/or ablated.

        The applied factor is `strength * max_act` of the latent (or `factor`
        verbatim), echoed back as `steering_factor`; `formatted_prompt` and
        `prompt_tokens` let a client measure the continuation on exactly what
        the model saw. `ablated_latent` echoes the projected-out latent.
        """
        if request.ablate is not None:
            if request.ablate >= crosscoder.dict_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Latent {request.ablate} out of range "
                    f"[0, {crosscoder.dict_size})",
                )
            if request.ablate == request.latent:
                raise HTTPException(
                    status_code=400,
                    detail="Steering and ablating the same latent contradict each other",
                )
        if request.latent is None:
            if request.strength is not None or request.factor is not None:
                raise HTTPException(
                    status_code=400, detail="strength and factor require a latent"
                )
        elif (request.strength is None) == (request.factor is None):
            raise HTTPException(
                status_code=400,
                detail="give exactly one of strength and factor with a latent",
            )
        factor = None
        steering = None
        if request.latent is not None:
            if request.latent >= crosscoder.dict_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Latent {request.latent} out of range "
                    f"[0, {crosscoder.dict_size})",
                )
            factor = (
                request.factor
                if request.factor is not None
                else request.strength * max_acts[request.latent].item()
            )
            assert th.isfinite(
                th.tensor(factor)
            ), f"Latent {request.latent} has no finite steering factor"
            steering = SteeringSpec(
                vector=get_crosscoder_latent(
                    request.latent, crosscoder, layer=FT_SIDE
                ).clone(),
                strength=factor,
            )
        ablation = (
            None
            if request.ablate is None
            else AblationSpec(
                vector=get_crosscoder_latent(
                    request.ablate, crosscoder, layer=FT_SIDE
                ).clone()
            )
        )
        formatted = format_text(request.prompt, request.raw)
        return {
            "steering_factor": factor,
            "ablated_latent": request.ablate,
            "formatted_prompt": formatted,
            "prompt_tokens": len(
                backend.tokenizer(formatted, add_special_tokens=True)["input_ids"]
            ),
            "text": _generate(formatted, steering, ablation, request),
        }

    return app
