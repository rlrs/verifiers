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

A turn the engine ended by repetition detection (finish reason `repetition`) is judged on
the tokens before its loop: the repeating tail is stripped first (`loop_start`), since a
low-NLL loop would otherwise dilute a derailed prefix, and the cut itself would otherwise
change the turn's mean.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

REPETITION = "repetition"
"""vLLM's finish reason for a `SamplingParams.repetition_detection` stop."""

DEFAULT_PATTERN_SIZES = (16, 64)
"""(min, max) loop period searched when stripping a repetition-stopped turn's tail."""

DEGENERATE_TURN = "degenerate_turn"
"""The trace stop condition set when the live check fires."""


@dataclass(frozen=True)
class DegenerateTurn:
    index: int
    """Position among the trace's sampled nodes (0 = first sampled turn)."""
    node: int
    """Index into `Trace.nodes`."""
    tokens: int
    """Tokens judged (for a repetition stop, the tokens before the loop)."""
    mean_nll: float


def loop_start(
    tokens: Sequence[int], min_pattern_size: int, max_pattern_size: int
) -> int:
    """Where the periodic tail of `tokens` begins (`len(tokens)` when there is none).

    Like vLLM's repetition check, it takes the shortest period `p` in
    [`min_pattern_size`, `max_pattern_size`] whose tail repeats (the last `p` tokens equal
    the `p` before them), then extends the periodic run back as far as
    `tokens[i] == tokens[i + p]` holds. Everything from there on — the loop's first copy,
    every full repetition and the trailing partial one — is the loop."""
    n = len(tokens)
    for p in range(max(1, min_pattern_size), max_pattern_size + 1):
        if 2 * p > n:
            break
        if all(tokens[n - k] == tokens[n - p - k] for k in range(1, p + 1)):
            start = n - 2 * p
            while start > 0 and tokens[start - 1] == tokens[start - 1 + p]:
                start -= 1
            return start
    return n


def turn_mean_nll(
    logprobs: Sequence[Any] | None,
    finish_reason: str | None,
    *,
    min_tokens: int,
    max_mean_nll: float,
    token_ids: Sequence[int] | None = None,
    pattern_sizes: tuple[int, int] = DEFAULT_PATTERN_SIZES,
) -> tuple[int, float] | None:
    """`(tokens, mean_nll)` when one sampled turn is degenerate, else None.

    A turn is degenerate when it has at least `min_tokens` numeric logprobs and their mean
    negative value exceeds `max_mean_nll`. `token_ids` are the turn's sampled tokens,
    aligned with `logprobs`; for a `repetition` stop the rule applies to the logprobs
    before `loop_start(token_ids, *pattern_sizes)`."""
    logprobs = list(logprobs or ())
    if (
        finish_reason == REPETITION
        and token_ids is not None
        and len(token_ids) == len(logprobs)
    ):
        logprobs = logprobs[: loop_start(token_ids, *pattern_sizes)]
    values = [lp for lp in logprobs if isinstance(lp, (int, float))]
    if len(values) < min_tokens:
        return None
    mean_nll = -sum(values) / len(values)
    if mean_nll > max_mean_nll:
        return len(values), mean_nll
    return None


def sampled_token_ids(node: Any) -> list[int]:
    """A message node's sampled tokens (`token_ids` under `mask`), aligned with `logprobs`."""
    token_ids = getattr(node, "token_ids", None) or []
    mask = getattr(node, "mask", None) or []
    return [token for token, sampled in zip(token_ids, mask) if sampled]


def node_mean_nll(
    node: Any,
    finish_reason: str | None,
    *,
    min_tokens: int,
    max_mean_nll: float,
    pattern_sizes: tuple[int, int] = DEFAULT_PATTERN_SIZES,
) -> tuple[int, float] | None:
    """`turn_mean_nll` over one committed sampled node (`logprobs`, `token_ids`, `mask`) —
    the one entry point the live check and the post-hoc scan both use."""
    return turn_mean_nll(
        getattr(node, "logprobs", None),
        finish_reason,
        min_tokens=min_tokens,
        max_mean_nll=max_mean_nll,
        token_ids=sampled_token_ids(node) if finish_reason == REPETITION else None,
        pattern_sizes=pattern_sizes,
    )


def node_finish_reasons(calls: Iterable[Any] | None) -> dict[int, str | None]:
    """Committed node index -> the finish reason of the model call that produced it."""
    out: dict[int, str | None] = {}
    for call in calls or ():
        node = getattr(call, "node", None)
        if node is not None and getattr(call, "error", None) is None:
            out[node] = getattr(call, "finish_reason", None)
    return out


def find_degenerate_turn(
    trace: Any,
    *,
    min_tokens: int,
    max_mean_nll: float,
    pattern_sizes: tuple[int, int] = DEFAULT_PATTERN_SIZES,
) -> DegenerateTurn | None:
    """The first degenerate sampled turn of a (finished) trace, or None.

    Duck-typed over `trace.nodes` (`sampled`, `logprobs`, `token_ids`, `mask`) and
    `trace.calls` (`node`, `finish_reason`), so recorded traces loaded as plain objects
    work too."""
    finish = node_finish_reasons(getattr(trace, "calls", None))
    index = 0
    for position, node in enumerate(getattr(trace, "nodes", None) or ()):
        if not getattr(node, "sampled", False):
            continue
        verdict = node_mean_nll(
            node,
            finish.get(position),
            min_tokens=min_tokens,
            max_mean_nll=max_mean_nll,
            pattern_sizes=pattern_sizes,
        )
        if verdict is not None:
            return DegenerateTurn(
                index=index, node=position, tokens=verdict[0], mean_nll=verdict[1]
            )
        index += 1
    return None
