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
        "--queue",
        type=int,
        default=32,
        metavar="N",
        help="how many requests may WAIT for the GPU before the server starts refusing "
        "with 503 (the one being served is not counted; 0 = serve one, refuse the rest). "
        "A refusal a client can retry beats a wait it cannot predict.",
    )
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
    from diffing.serving.launch import serve_crosscoder

    args = parse_args(build_parser())

    overrides = list(args.overrides)
    if args.config_dirs is None and not any(
        override.startswith("diffing/method=") for override in overrides
    ):
        overrides.insert(0, "diffing/method=crosscoder")

    cfg = compose_config(args.config_dirs, args.config_name, overrides)
    serve_crosscoder(cfg, host=args.host, port=args.port, queue=args.queue)


if __name__ == "__main__":
    main()
