"""Where a GEPA run writes: mirrors `verifiers.v1.cli.output.output_path`, with a `gepa/`
segment so optimization runs don't share a directory namespace with eval runs of the same
taskset/model/harness."""

from pathlib import Path

from verifiers.v1.configs.gepa import GEPAConfig


def gepa_output_path(config: GEPAConfig) -> Path:
    """`outputs/gepa/<taskset>--<model>--<harness>/<uuid>` (or the explicit `--run-dir`). The
    per-run `uuid` leaf means runs never overwrite each other."""
    if config.run_dir is not None:
        return config.run_dir
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / "gepa" / name / config.uuid
