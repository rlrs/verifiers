"""The GEPA entrypoint: `uv run gepa [<taskset-id>] --model <model> [options]`.

Registered as the `gepa` console script. Optimizes a v1 taskset's `Task.system_prompt` via
GEPA (Genetic-Pareto): alternating rollouts with a teacher LM reflecting on results — see
`verifiers.v1.cli.gepa.adapter.GEPAv1Adapter`. Mirrors the `eval` CLI's taskset/harness
resolution (`cli/resolve.py`). v1-native tasksets only — a legacy (v0) env / the env-server
worker pool aren't supported here; run those through the existing `vf-gepa` command instead.

`-h`/`--help` (or no args) prints the local example tasksets/harnesses plus the full, typed
pydantic-config help — narrowed to whatever `--taskset.id` / `--harness.id` are given.
"""

import logging
import signal
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.gepa.output import gepa_output_path
from verifiers.v1.cli.gepa.runner import run_gepa
from verifiers.v1.cli.output import write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.gepa import GEPAv1Config
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run gepa [<taskset-id>] --model <model> [--harness.id <id>] [options] [@ file.toml]\n"
    "       (legacy v0 environments run through the existing `vf-gepa` command instead)"
)


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(
            narrow_config(GEPAv1Config, argv)
        )  # full option help, narrowed to the given ids
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    config_type = narrow_config(GEPAv1Config, argv)
    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    if config.is_legacy:
        raise SystemExit(
            f"{USAGE}\nrun a legacy (v0) environment through `vf-gepa` instead of `gepa`"
        )
    setup_logging("DEBUG" if config.verbose else "INFO")
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        logger.info(
            "wrote config to %s", write_config(config, gepa_output_path(config))
        )
        return

    # Make SIGTERM behave like Ctrl-C (SIGINT) so a killed/timed-out run still runs the
    # adapter's `serving()` teardown (tears down interception pool / tool-server runtimes).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    env = vf.Environment(config)
    result = run_gepa(env, config)
    print(f"best system prompt:\n{result.best_candidate.get('system_prompt', '')}")
