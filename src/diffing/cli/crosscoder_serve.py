"""CLI entry point for the crosscoder compute server.

Serves one trained crosscoder over HTTP: per-token latent activations for a text,
and generation optionally steered by a latent's decoder direction.

Usage:
    crosscoder-serve --host 0.0.0.0 --port 8000 model=qwen3_1_7B organism=cake_bake
    crosscoder-serve --config-dir /path/to/run --config-name my_run --port 8000

Positional arguments are standard Hydra overrides, composed against the same
`configs/config.yaml` as `main.py`, and may be interleaved with the flags;
`diffing/method=crosscoder` is selected unless overridden, because the served
dictionary is identified by the crosscoder method's own run name.

With `--config-dir` the primary config is read from that directory instead
(`--config-name` names it), with the package `configs/` on the search path — the
form a generated run config uses (a file whose defaults extend `config` and
override model/organism). Repeat `--config-dir` to add further search-path
directories (extra model/organism configs).

The crosscoder is read from the directory the crosscoder method writes:
    <diffing.results_dir>/crosscoder/layer_<L>/<run name>/dictionary_model
so the method must have been run for this model/organism/layer first.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Command line: uvicorn binding, config selection, plus Hydra overrides."""
    parser = argparse.ArgumentParser(prog="crosscoder-serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--config-dir",
        action="append",
        dest="config_dirs",
        default=None,
        help="directory holding the primary config; repeatable, extra dirs join "
        "the search path",
    )
    parser.add_argument("--config-name", default="config")
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
    if args.config_dirs and any(
        token.startswith("hydra.searchpath=") for token in args.overrides
    ):
        parser.error(
            "with --config-dir the search path is managed by the CLI; "
            "repeat --config-dir instead of overriding hydra.searchpath"
        )
    return args


def compose_config(
    config_dirs: list[str] | None, config_name: str, overrides: list[str]
):
    """Compose the run config exactly as `main.py` would see it.

    Without `config_dirs` the package `configs/` is the primary config directory.
    With it, the first directory holds the primary config and the package
    `configs/` (plus any further directories) joins the Hydra search path, so a
    generated config's `defaults: [config, override model: …]` resolve.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from diffing.utils.configs import CONFIGS_DIR

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    if not config_dirs:
        with initialize_config_dir(
            config_dir=str(CONFIGS_DIR.resolve()), version_base=None
        ):
            return compose(config_name=config_name, overrides=overrides)
    primary = Path(config_dirs[0]).resolve()
    assert primary.is_dir(), f"--config-dir {primary} is not a directory"
    search = [f"file://{CONFIGS_DIR.resolve()}"] + [
        f"file://{Path(d).resolve()}" for d in config_dirs[1:]
    ]
    with initialize_config_dir(config_dir=str(primary), version_base=None):
        return compose(
            config_name=config_name,
            overrides=overrides + [f"hydra.searchpath=[{','.join(search)}]"],
        )


def main() -> None:
    import torch as th
    from loguru import logger
    import uvicorn

    from diffing.serving.backend import ClassicBackend
    from diffing.serving.server import build_app
    from diffing.utils.activations import get_layer_indices
    from diffing.utils.configs import get_model_configurations
    from diffing.utils.dictionary.training import (
        crosscoder_results_dir,
        crosscoder_run_name,
    )
    from diffing.utils.dictionary.utils import load_dictionary_model, load_latent_df

    args = parse_args(build_parser())

    overrides = list(args.overrides)
    if args.config_dirs is None and not any(
        override.startswith("diffing/method=") for override in overrides
    ):
        overrides.insert(0, "diffing/method=crosscoder")

    cfg = compose_config(args.config_dirs, args.config_name, overrides)

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
