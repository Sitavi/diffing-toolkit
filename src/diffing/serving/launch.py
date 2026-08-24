"""Serving one trained crosscoder from a composed config, for the CLI and for the pipeline."""

from pathlib import Path

import torch as th
import uvicorn
from loguru import logger
from omegaconf import DictConfig

from diffing.serving.backend import ClassicBackend
from diffing.serving.queue import GpuQueue
from diffing.serving.server import build_app
from diffing.utils.activations import get_layer_indices
from diffing.utils.configs import get_model_configurations
from diffing.utils.dictionary.training import (
    crosscoder_results_dir,
    crosscoder_run_name,
)
from diffing.utils.dictionary.utils import load_dictionary_model, load_latent_df

SERVE_AFTER = "serve_after"
SERVE_AFTER_KEYS = {"enabled", "host", "port", "queue"}
SERVE_AFTER_MODES = ("full", "diffing", "no_evaluation")


def serve_crosscoder(cfg: DictConfig, host: str, port: int, queue: int) -> None:
    """Serve over HTTP the crosscoder this config identifies, until the process is stopped.

    The dictionary is read from the directory the crosscoder method writes, so that method must
    have run for this model, organism and layer first. A config naming another method, naming
    more than one layer, or pointing at a directory holding no trained crosscoder raises.
    """
    assert (
        cfg.diffing.method.name == "crosscoder"
    ), f"crosscoder-serve serves crosscoders, got method {cfg.diffing.method.name}"
    th.set_float32_matmul_precision(cfg.torch_precision)

    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)
    layers = cfg.diffing.method.layers
    if layers is None:
        layers = cfg.preprocessing.layers
    assert len(layers) == 1, f"crosscoder-serve serves exactly one layer, got {layers}"
    layer = get_layer_indices(base_model_cfg.model_id, layers)[0]

    dictionary_name = crosscoder_run_name(
        cfg, layer, base_model_cfg, finetuned_model_cfg
    )
    dictionary_dir = (
        crosscoder_results_dir(Path(cfg.diffing.results_dir), layer, dictionary_name)
        / "dictionary_model"
    )
    assert (
        dictionary_dir.exists()
    ), f"No trained crosscoder at {dictionary_dir}, run the crosscoder method first"

    latent_df_source = next(
        (
            directory
            for directory in (dictionary_dir, dictionary_dir.parent)
            if (directory / "latent_df.csv").is_file()
        ),
        dictionary_name,
    )
    latent_df = load_latent_df(latent_df_source)
    max_act_column = (
        "max_act_validation"
        if "max_act_validation" in latent_df.columns
        else "max_act_train"
    )
    max_acts = th.tensor(latent_df[max_act_column].to_numpy(), dtype=th.float32)

    logger.info(f"Serving {dictionary_name} (layer {layer}) from {dictionary_dir}")
    backend = ClassicBackend.from_config(cfg, layer)
    crosscoder = load_dictionary_model(dictionary_dir, is_sae=False).to(backend.device)
    assert crosscoder.activation_dim == backend.models["base"].hidden_size, (
        f"Crosscoder activation_dim {crosscoder.activation_dim} does not match "
        f"model hidden_size {backend.models['base'].hidden_size}"
    )
    app = build_app(backend, crosscoder, cfg, max_acts, queue=GpuQueue(capacity=queue))

    logger.info(f"Listening on {host}:{port} (one GPU, FIFO, up to {queue} waiting)")
    uvicorn.run(app, host=host, port=port)


def serve_after_binding(cfg: DictConfig) -> dict | None:
    """The host, port and queue depth a run asks the pipeline to serve on once the diffing stage
    is done, None when the method declares no block or the block is disabled; an incomplete
    block raises."""
    binding = cfg.diffing.method.get(SERVE_AFTER)
    if binding is None:
        return None
    missing = sorted(SERVE_AFTER_KEYS - set(binding))
    assert not missing, f"diffing.method.{SERVE_AFTER} lacks {missing}"
    if not binding.enabled:
        return None
    return {
        "host": binding.host,
        "port": int(binding.port),
        "queue": int(binding.queue),
    }


def serve_after_diffing(cfg: DictConfig, mode: str) -> bool:
    """Serve the crosscoder the pipeline has just trained and analysed when the config asks for
    it and ``mode`` ran the diffing stage, blocking until the process is stopped; returns whether
    the server ran."""
    if mode not in SERVE_AFTER_MODES:
        return False
    binding = serve_after_binding(cfg)
    if binding is None:
        return False
    logger.info("serve_after: the diffing stage is done, starting the compute server")
    serve_crosscoder(cfg, **binding)
    return True
