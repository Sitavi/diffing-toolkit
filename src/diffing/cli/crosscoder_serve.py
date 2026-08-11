"""CLI entry point for the crosscoder compute server.

Serves one trained crosscoder over HTTP: per-token latent activations for a text,
and generation optionally steered by a latent's decoder direction.

Usage:
    crosscoder-serve --host 0.0.0.0 --port 8000 model=qwen3_1_7B organism=cake_bake

Positional arguments are standard Hydra overrides, composed against the same
`configs/config.yaml` as `main.py`, and may be interleaved with the flags;
`diffing/method=crosscoder` is selected unless overridden, because the served
dictionary is identified by the crosscoder method's own run name.

The crosscoder is read from the directory the crosscoder method writes:
    <diffing.results_dir>/crosscoder/layer_<L>/<run name>/dictionary_model
so the method must have been run for this model/organism/layer first.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Command line: uvicorn binding plus Hydra overrides."""
    parser = argparse.ArgumentParser(prog="crosscoder-serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. model=qwen3_1_7B organism=cake_bake",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    """Parse flags plus Hydra overrides, wherever they appear on the line.

    argparse stops filling a `nargs='*'` positional at the first flag, so
    overrides after `--host`/`--port` land in the unknown bucket; anything
    unknown that is not an override is still an error.
    """
    args, unknown = parser.parse_known_args(argv)
    for token in unknown:
        if "=" not in token or token.startswith("-"):
            parser.error(f"unrecognized argument: {token}")
    args.overrides = list(args.overrides) + unknown
    return args


def main() -> None:
    import torch as th
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from loguru import logger
    import uvicorn

    from diffing.serving.backend import ClassicBackend
    from diffing.serving.server import build_app
    from diffing.utils.activations import get_layer_indices
    from diffing.utils.configs import CONFIGS_DIR, get_model_configurations
    from diffing.utils.dictionary.training import (
        crosscoder_results_dir,
        crosscoder_run_name,
    )
    from diffing.utils.dictionary.utils import load_dictionary_model, load_latent_df

    args = parse_args(build_parser())

    overrides = list(args.overrides)
    if not any(override.startswith("diffing/method=") for override in overrides):
        overrides.insert(0, "diffing/method=crosscoder")

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR.resolve()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

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
    app = build_app(backend, crosscoder, cfg, max_acts)

    logger.info(f"Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
