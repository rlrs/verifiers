"""The degenerate-turn rule: one sampled turn whose tokens are near-random.

At temperature 1 with no tail cut-off, a rare sampled token can derail a long agent
context into thousands of near-random tokens. The sampler's own logprobs show it: a
normal turn averages well under 1 nat per token, a derailed one several.

This module is the single definition of the per-turn rule. Two callers must reach the
same verdict on the same data:

- live: the interception server checks each turn right after committing it and stops
  the rollout (`RolloutLimits.degenerate_turn`, stop condition `degenerate_turn`), so no
  tokens are spent on a trace that will be discarded;
- post hoc: a trainer (prime-rl's `find_degenerate_turn`) scans the finished trace and
  drops it from training.

A turn the engine ended by repetition detection (finish reason `repetition`) is exempt:
cutting a low-NLL loop raises the turn's mean NLL, and those turns should be treated
like a max-tokens truncation, not dropped as degenerate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

EXEMPT_FINISH_REASONS = frozenset({"repetition"})
"""Finish reasons whose turns the rule never judges (see the module docstring)."""

DEGENERATE_TURN = "degenerate_turn"
"""The trace stop condition set when the live check fires."""


@dataclass(frozen=True)
class DegenerateTurn:
    index: int
    """Position among the trace's sampled nodes (0 = first sampled turn)."""
    node: int
    """Index into `Trace.nodes`."""
    tokens: int
    mean_nll: float


def turn_mean_nll(
    logprobs: Sequence[Any] | None,
    finish_reason: str | None,
    *,
    min_tokens: int,
    max_mean_nll: float,
) -> tuple[int, float] | None:
    """`(tokens, mean_nll)` when one sampled turn is degenerate, else None.

    A turn is degenerate when it has at least `min_tokens` numeric logprobs and their mean
    negative value exceeds `max_mean_nll`, unless its finish reason is exempt."""
    if finish_reason in EXEMPT_FINISH_REASONS:
        return None
    values = [lp for lp in (logprobs or ()) if isinstance(lp, (int, float))]
    if len(values) < min_tokens:
        return None
    mean_nll = -sum(values) / len(values)
    if mean_nll > max_mean_nll:
        return len(values), mean_nll
    return None


def node_finish_reasons(calls: Iterable[Any] | None) -> dict[int, str | None]:
    """Committed node index -> the finish reason of the model call that produced it."""
    out: dict[int, str | None] = {}
    for call in calls or ():
        node = getattr(call, "node", None)
        if node is not None and getattr(call, "error", None) is None:
            out[node] = getattr(call, "finish_reason", None)
    return out


def find_degenerate_turn(
    trace: Any, *, min_tokens: int, max_mean_nll: float
) -> DegenerateTurn | None:
    """The first degenerate sampled turn of a (finished) trace, or None.

    Duck-typed over `trace.nodes` (`sampled`, `logprobs`) and `trace.calls` (`node`,
    `finish_reason`), so recorded traces loaded as plain objects work too."""
    finish = node_finish_reasons(getattr(trace, "calls", None))
    index = 0
    for position, node in enumerate(getattr(trace, "nodes", None) or ()):
        if not getattr(node, "sampled", False):
            continue
        verdict = turn_mean_nll(
            getattr(node, "logprobs", None),
            finish.get(position),
            min_tokens=min_tokens,
            max_mean_nll=max_mean_nll,
        )
        if verdict is not None:
            return DegenerateTurn(
                index=index, node=position, tokens=verdict[0], mean_nll=verdict[1]
            )
        index += 1
    return None
