"""The outcome a GEPA run writes to its output dir (which `verifiers.v1.cli.output.output_path`
resolves, shared with eval). v1-native: stdlib only, no v0 code — GEPA also persists its own
optimizer state (`gepa_state.bin`, logs) into the same dir."""

import json
from pathlib import Path

from gepa.core.result import GEPAResult

from verifiers.v1.gepa.config import GEPAConfig


def write_gepa_result(run_dir: Path, result: GEPAResult, config: GEPAConfig) -> None:
    """Write the run's outcome to `run_dir`: the optimized system prompt as plain text
    (`system_prompt.txt` — the deliverable) plus a JSON summary (`metadata.json`)."""
    best_prompt = (result.best_candidate or {}).get("system_prompt", "")
    (run_dir / "system_prompt.txt").write_text(best_prompt, encoding="utf-8")

    scores = list(getattr(result, "val_aggregate_scores", None) or [])
    best_idx = getattr(result, "best_idx", None)
    best_score = (
        scores[best_idx]
        if best_idx is not None and 0 <= best_idx < len(scores)
        else None
    )
    metadata = {
        "env_id": config.taskset.id,
        "model": config.model,
        "reflection_model": config.reflection_model or config.model,
        "num_train": config.num_train,
        "num_val": config.num_val,
        "max_metric_calls": config.max_metric_calls,
        "best_idx": best_idx,
        "best_val_score": best_score,
        "num_candidates": len(getattr(result, "candidates", None) or []),
        "system_prompt": best_prompt,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
