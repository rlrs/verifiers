"""The GEPA <-> v1 bridge: run a candidate system prompt over a batch of tasks and score it.

GEPA's adapter protocol (`evaluate`, `make_reflective_dataset`) is synchronous, and
`gepa.api.optimize` itself blocks — but `Environment.serving(tasks)` is an async context
manager that must stay open (and pinned to one event loop) for the life of the run. So
`GEPAv1Adapter` owns a single persistent loop: entered once (as a context manager) around the
whole `optimize()` call, and reused by every `evaluate()` batch — one `run_until_complete` per
call, exactly like v0's adapter, plus the one long-lived `serving()` bracket v1 needs.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch

from verifiers.utils.save_utils import make_serializable
from verifiers.v1.clients import ModelContext
from verifiers.v1.env import Environment
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

if TYPE_CHECKING:
    from verifiers.gepa.display import GEPADisplay

Candidate = dict[str, str]


@dataclass
class GEPAv1Adapter:
    """Bridges GEPA's optimization loop with a native v1 `Environment`. `tasks` covers only
    the trainset + valset tasks GEPA was given (not the whole taskset), keyed by `Task.idx` —
    GEPA's `batch` is a list of those idxs, and injecting the candidate is a `model_copy` on
    each looked-up (frozen) `Task`."""

    env: Environment
    ctx: ModelContext
    tasks: dict[int, Task]
    max_concurrent: int | None = 32
    state_columns: list[str] = field(default_factory=list)
    display: "GEPADisplay | None" = None
    propose_new_texts: Callable[..., Candidate] | None = None
    """Part of GEPA's adapter protocol — its proposer reads this attribute on every reflection
    step. None = use GEPA's default reflection-LM proposer (the AttributeError from leaving it
    undeclared silently disables all mutation proposals)."""

    loop: asyncio.AbstractEventLoop = field(
        default_factory=asyncio.new_event_loop, repr=False
    )
    semaphore: asyncio.Semaphore | None = field(default=None, repr=False, init=False)
    _serving: contextlib.AbstractAsyncContextManager | None = field(
        default=None, repr=False, init=False
    )
    _seen_prompts: dict[str, int] = field(default_factory=dict, repr=False)

    def __enter__(self) -> "GEPAv1Adapter":
        self.semaphore = (
            asyncio.Semaphore(self.max_concurrent) if self.max_concurrent else None
        )
        self._serving = self.env.serving(list(self.tasks.values()))
        self.loop.run_until_complete(self._serving.__aenter__())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._serving is not None
        self.loop.run_until_complete(self._serving.__aexit__(exc_type, exc, tb))
        self.loop.close()

    def evaluate(
        self,
        batch: list[int],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trace, Trace]:
        """Run `candidate`'s system prompt on the tasks named by `batch` (`Task.idx` values)
        and score them."""
        system_prompt = candidate.get("system_prompt", "")
        traces = self.loop.run_until_complete(self._run_batch(batch, system_prompt))
        scores = [trace.reward for trace in traces]

        if self.display is not None:
            candidate_idx = self._seen_prompts.setdefault(
                system_prompt, len(self._seen_prompts)
            )
            self.display.update_eval(
                candidate_idx=candidate_idx,
                scores=scores,
                example_ids=list(batch),
                capture_traces=capture_traces,
            )

        return EvaluationBatch(
            outputs=traces,
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    async def _run_batch(self, batch: list[int], system_prompt: str) -> list[Trace]:
        tasks = [
            self.tasks[idx].model_copy(update={"system_prompt": system_prompt})
            for idx in batch
        ]
        episodes = [self.env.episode(task, self.ctx, n=1) for task in tasks]
        results = await asyncio.gather(
            *(episode.run(self.semaphore) for episode in episodes)
        )
        return [trace for episode_traces in results for trace in episode_traces]

    def make_reflective_dataset(
        self,
        candidate: Candidate,  # noqa: ARG002 - required by GEPA's adapter protocol
        eval_batch: EvaluationBatch[Trace, Trace],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build the reflective dataset the teacher LM reads to propose a new system prompt,
        from `eval_batch.trajectories` (traces captured by a prior `evaluate(capture_traces=True)`
        on the same batch)."""
        traces = eval_batch.trajectories or []
        records = []
        for trace, score in zip(traces, eval_batch.scores):
            record: dict[str, Any] = {
                "query": trace.task.prompt_text,
                "completion": trace.last_reply,
                "reward": score,
            }
            if trace.has_error:
                record["error"] = str(trace.error)
            if trace.stop_condition:
                record["stop_condition"] = trace.stop_condition
            for col in self.state_columns:
                if col in trace.info:
                    record[col] = make_serializable(trace.info[col])
                elif hasattr(trace.task, col):
                    record[col] = make_serializable(getattr(trace.task, col))
            records.append(record)
        return {comp: records for comp in components_to_update}
