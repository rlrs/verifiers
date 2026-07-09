"""GEPA-for-v1 tests: the train/val split + seed-prompt validation, the `GEPAv1Adapter`
(against a fake `Environment` — no real rollouts), `GEPAConfig` defaults and the CLI's
argument guards, and one `@pytest.mark.e2e` live optimization run (needs `PRIME_API_KEY`;
skipped without one, same as the rest of the v1 e2e suite — see `conftest.py`)."""

import contextlib
import logging

import pytest
from gepa.core.adapter import EvaluationBatch
from pydantic import ValidationError

from verifiers.v1.cli.gepa.adapter import GEPAv1Adapter
from verifiers.v1.cli.gepa.dataset import resolve_gepa_seed_prompt, split_tasks
from verifiers.v1.cli.gepa.main import main
from verifiers.v1.cli.gepa.runner import run_gepa
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.gepa import GEPAConfig
from verifiers.v1.env import Environment
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace
from verifiers.v1.types import SamplingConfig

# --- split_tasks + resolve_gepa_seed_prompt -----------------------------------


def make_tasks(n: int, system_prompt: str | None = None) -> list[Task]:
    return [
        Task(idx=i, prompt=f"task {i}", system_prompt=system_prompt) for i in range(n)
    ]


class FakeHarnessConfig:
    id = "fake-harness"


class FakeHarness:
    APPENDS_SYSTEM_PROMPT = False

    def __init__(self, appends: bool) -> None:
        self.APPENDS_SYSTEM_PROMPT = (
            appends  # instance override, mirrors a real ClassVar
        )
        self.config = FakeHarnessConfig()


class FakeSeedEnv:
    def __init__(self, appends: bool = True) -> None:
        self.harness = FakeHarness(appends)


def test_split_tasks_is_disjoint_and_sized():
    tasks = make_tasks(10)
    train, val = split_tasks(tasks, num_train=6, num_val=3, shuffle=False, seed=0)
    assert len(train) == 6
    assert len(val) == 3
    assert {t.idx for t in train}.isdisjoint({t.idx for t in val})


def test_split_tasks_shuffle_is_reproducible_with_seed():
    tasks = make_tasks(20)
    train_a, val_a = split_tasks(tasks, num_train=5, num_val=5, shuffle=True, seed=7)
    train_b, val_b = split_tasks(tasks, num_train=5, num_val=5, shuffle=True, seed=7)
    assert [t.idx for t in train_a] == [t.idx for t in train_b]
    assert [t.idx for t in val_a] == [t.idx for t in val_b]


def test_split_tasks_raises_when_split_exceeds_pool():
    tasks = make_tasks(3)
    with pytest.raises(ValueError, match="only loaded 3"):
        split_tasks(tasks, num_train=2, num_val=2, shuffle=False, seed=0)


def test_resolve_seed_prompt_warns_on_non_appending_harness():
    """A non-appending harness folds the candidate into the user prompt — GEPA still runs, so
    this warns and proceeds rather than blocking."""
    env = FakeSeedEnv(appends=False)
    tasks = make_tasks(2, system_prompt="be helpful")
    records: list[logging.LogRecord] = []
    logger = logging.getLogger("verifiers.v1.cli.gepa.dataset")
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        seed = resolve_gepa_seed_prompt(env, tasks, None)
    finally:
        logger.removeHandler(handler)
    assert seed == "be helpful"
    assert any("APPENDS_SYSTEM_PROMPT" in r.getMessage() for r in records)


def test_resolve_seed_prompt_prefers_explicit_initial_prompt():
    env = FakeSeedEnv(appends=True)
    tasks = make_tasks(2, system_prompt="from taskset")
    assert resolve_gepa_seed_prompt(env, tasks, "from cli") == "from cli"


def test_resolve_seed_prompt_falls_back_to_first_task_system_prompt():
    env = FakeSeedEnv(appends=True)
    tasks = make_tasks(2, system_prompt="from taskset")
    assert resolve_gepa_seed_prompt(env, tasks, None) == "from taskset"


def test_resolve_seed_prompt_raises_when_no_task_sets_system_prompt():
    env = FakeSeedEnv(appends=True)
    tasks = make_tasks(2, system_prompt=None)
    with pytest.raises(ValueError, match="no task in this taskset"):
        resolve_gepa_seed_prompt(env, tasks, None)


# --- GEPAv1Adapter -------------------------------------------------------------


class FakeEpisode:
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    async def run(self, semaphore=None) -> list[Trace]:
        return [self.trace]


class FakeEnv:
    """Records the (candidate-injected) task each `episode()` call receives, and returns one
    canned `Trace` per task, scored by its idx."""

    def __init__(self, reward_by_idx: dict[int, float]) -> None:
        self.reward_by_idx = reward_by_idx
        self.seen_tasks: list[Task] = []

    def episode(self, task: Task, ctx: ModelContext, n: int = 1) -> FakeEpisode:
        self.seen_tasks.append(task)
        trace = Trace(task=task, rewards={"r": self.reward_by_idx[task.idx]})
        return FakeEpisode(trace)

    @contextlib.asynccontextmanager
    async def serving(self, tasks: list[Task]):
        yield


def make_adapter(
    tasks: dict[int, Task], reward_by_idx: dict[int, float]
) -> tuple[GEPAv1Adapter, FakeEnv]:
    env = FakeEnv(reward_by_idx)
    ctx = ModelContext(client=object(), model="test-model", sampling=SamplingConfig())
    adapter = GEPAv1Adapter(env=env, ctx=ctx, tasks=tasks, max_concurrent=4)
    return adapter, env


def test_evaluate_injects_candidate_system_prompt_and_scores_by_reward():
    tasks = {0: Task(idx=0, prompt="p0"), 1: Task(idx=1, prompt="p1")}
    adapter, env = make_adapter(tasks, reward_by_idx={0: 1.0, 1: 0.5})

    with adapter:
        batch = adapter.evaluate([0, 1], {"system_prompt": "be concise"})

    assert [t.system_prompt for t in env.seen_tasks] == ["be concise", "be concise"]
    assert batch.scores == [1.0, 0.5]
    assert batch.trajectories is None  # capture_traces defaults to False


def test_evaluate_captures_trajectories_when_requested():
    tasks = {0: Task(idx=0, prompt="p0")}
    adapter, _ = make_adapter(tasks, reward_by_idx={0: 1.0})

    with adapter:
        batch = adapter.evaluate([0], {"system_prompt": "x"}, capture_traces=True)

    assert batch.trajectories == batch.outputs


def test_adapter_declares_propose_new_texts():
    """GEPA's proposer reads `adapter.propose_new_texts` on every reflection step; an adapter
    without the attribute raises AttributeError there, which GEPA swallows as "did not propose
    a new candidate" — silently disabling all prompt mutation."""
    adapter, _ = make_adapter({}, reward_by_idx={})
    assert adapter.propose_new_texts is None


def test_make_reflective_dataset_builds_one_record_per_trace():
    task = Task(idx=0, prompt="what is 2+2?")
    trace = Trace(task=task, rewards={"r": 0.0})
    trace.errors.append(Error(type="ValueError", message="boom"))
    trace.stop_condition = "error"
    trace.info["notes"] = "scratch"

    adapter, _ = make_adapter({0: task}, reward_by_idx={0: 0.0})
    adapter.state_columns = ["notes"]
    eval_batch = EvaluationBatch(outputs=[trace], scores=[0.0], trajectories=[trace])

    dataset = adapter.make_reflective_dataset(
        {"system_prompt": "x"}, eval_batch, ["system_prompt"]
    )

    (record,) = dataset["system_prompt"]
    assert record["query"] == "what is 2+2?"
    assert record["completion"] == ""  # no assistant messages in this fake trace
    assert record["reward"] == 0.0
    assert "boom" in record["error"]
    assert record["stop_condition"] == "error"
    assert record["notes"] == "scratch"


# --- GEPAConfig + CLI guards --------------------------------------------------


def test_model_is_required():
    with pytest.raises(ValidationError):
        GEPAConfig(taskset={"id": "echo-v1"})


def test_defaults():
    config = GEPAConfig(taskset={"id": "echo-v1"}, model="gpt-4.1-mini")
    assert config.num_train == 100
    assert config.num_val == 50
    assert config.shuffle is True
    assert config.max_metric_calls == 500
    assert config.reflection_minibatch_size == 3
    assert config.max_concurrent == 32
    assert config.save_results is True
    assert config.reflection_model is None
    assert config.initial_prompt is None


def test_is_legacy_inherited_from_env_config():
    v1_config = GEPAConfig(taskset={"id": "echo-v1"}, model="gpt-4.1-mini")
    assert v1_config.is_legacy is False
    legacy_config = GEPAConfig(id="some-v0-env", model="gpt-4.1-mini")
    assert legacy_config.is_legacy is True


def test_main_requires_a_taskset():
    with pytest.raises(SystemExit) as exc_info:
        main(["--model", "gpt-4.1-mini"])
    assert "usage: uv run gepa" in str(exc_info.value)


def test_main_rejects_legacy_env_id():
    with pytest.raises(SystemExit) as exc_info:
        main(["--id", "some-v0-env", "--model", "gpt-4.1-mini"])
    assert "vf-gepa" in str(exc_info.value)


def test_main_dry_run_writes_config(tmp_path):
    run_dir = tmp_path / "run"
    main(
        [
            "echo-v1",
            "--model",
            "gpt-4.1-mini",
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ]
    )
    assert (run_dir / "config.toml").exists()


# --- end-to-end ------------------------------------------------------------------

E2E_MODEL = "deepseek/deepseek-v4-flash"


@pytest.mark.e2e
def test_gepa_optimizes_echo_v1(tmp_path):
    """`echo-v1` has 3 tasks and already sets `Task.system_prompt` — a tiny 2/1 train/val split
    with a handful of metric calls is enough to exercise the whole loop (seed -> evaluate ->
    reflect -> re-evaluate -> save) without needing a real dataset download."""
    config = GEPAConfig(
        taskset={"id": "echo-v1"},
        harness={"id": "null"},
        model=E2E_MODEL,
        reflection_model=E2E_MODEL,
        num_train=2,
        num_val=1,
        max_metric_calls=8,
        reflection_minibatch_size=1,
        max_turns=2,
        sampling={"max_tokens": 512, "temperature": 0},
        timeout={"rollout": 180, "scoring": 60},
        run_dir=tmp_path,
    )
    env = Environment(config)

    result = run_gepa(env, config)

    assert isinstance(result.best_candidate.get("system_prompt"), str)
    assert (tmp_path / "config.toml").exists()
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "system_prompt.txt").exists()
    assert (tmp_path / "metadata.json").exists()
