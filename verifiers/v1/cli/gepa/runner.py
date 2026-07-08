"""The GEPA runner: split tasks, drive `gepa.api.optimize` against a `GEPAv1Adapter`, save
results. Synchronous (unlike `run_eval`) — GEPA's own adapter protocol is synchronous, so all
async v1 work happens inside the adapter's persistent event loop (see `GEPAv1Adapter`)."""

import logging

from gepa.api import optimize
from gepa.core.result import GEPAResult

from verifiers.gepa.display import GEPADisplay
from verifiers.gepa.gepa_utils import save_gepa_results
from verifiers.v1.cli.gepa.adapter import GEPAv1Adapter
from verifiers.v1.cli.gepa.dataset import resolve_gepa_seed_prompt, split_tasks
from verifiers.v1.cli.gepa.output import gepa_output_path
from verifiers.v1.cli.gepa.reflection import build_reflection_lm
from verifiers.v1.cli.output import write_config
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.configs.gepa import GEPAv1Config
from verifiers.v1.env import Environment

logger = logging.getLogger(__name__)


def run_gepa(env: Environment, config: GEPAv1Config) -> GEPAResult:
    logger.info("gepa config:\n%s", config.model_dump_json(indent=2))
    all_tasks = env.taskset.load_tasks()
    train_tasks, val_tasks = split_tasks(
        all_tasks, config.num_train, config.num_val, config.shuffle, config.seed
    )
    selected_tasks = [*train_tasks, *val_tasks]
    # Seed from the tasks GEPA actually evaluates (train ∪ val), not the full pre-split pool —
    # a taskset with per-task system prompts could otherwise seed from a task in neither split.
    seed_prompt = resolve_gepa_seed_prompt(env, selected_tasks, config.initial_prompt)
    tasks_by_idx = {task.idx: task for task in selected_tasks}

    run_dir = gepa_output_path(config) if config.save_results else None
    if run_dir is not None:
        write_config(config, run_dir)
        logger.info("results: %s", run_dir)

    display = GEPADisplay(
        env_id=env.taskset.config.id or env.taskset.name,
        model=config.model,
        reflection_model=config.reflection_model or config.model,
        max_metric_calls=config.max_metric_calls,
        num_train=len(train_tasks),
        num_val=len(val_tasks),
        log_file=run_dir / "gepa.log" if run_dir is not None else None,
        perfect_score=config.perfect_score,
    )
    # Tell the display the real valset ids/size so it classifies full-valset evals correctly
    # (it otherwise assumes the default size of 50 and mislabels the live UI for other --num-val).
    display.set_valset_info(len(val_tasks), [task.idx for task in val_tasks])

    with display:
        client = resolve_client(config.client)
        ctx = ModelContext(client=client, model=config.model, sampling=config.sampling)
        adapter = GEPAv1Adapter(
            env=env,
            ctx=ctx,
            tasks=tasks_by_idx,
            max_concurrent=config.max_concurrent,
            state_columns=config.state_columns,
            display=display,
        )
        reflection_lm = build_reflection_lm(config)

        with adapter:
            optimize_kwargs: dict = dict(
                seed_candidate={"system_prompt": seed_prompt},
                trainset=[task.idx for task in train_tasks],
                valset=[task.idx for task in val_tasks],
                adapter=adapter,
                reflection_lm=reflection_lm,
                max_metric_calls=config.max_metric_calls,
                reflection_minibatch_size=config.reflection_minibatch_size,
                run_dir=str(run_dir) if run_dir is not None else None,
                seed=config.seed,
                display_progress_bar=False,
                skip_perfect_score=config.perfect_score is not None,
                logger=display,
            )
            if config.perfect_score is not None:
                optimize_kwargs["perfect_score"] = config.perfect_score
            try:
                result = optimize(**optimize_kwargs)
            finally:
                # Close the client on the adapter's loop while it's still open — `__exit__`
                # closes that loop, so this must run even if `optimize` raises.
                adapter.loop.run_until_complete(client.close())

        save_path = None
        if run_dir is not None:
            save_gepa_results(
                run_dir,
                result,
                config={
                    "env_id": env.taskset.config.id,
                    "model": config.model,
                    "reflection_model": config.reflection_model or config.model,
                    "num_train": len(train_tasks),
                    "num_val": len(val_tasks),
                    "max_metric_calls": config.max_metric_calls,
                    "reflection_minibatch_size": config.reflection_minibatch_size,
                    "perfect_score": config.perfect_score,
                    "state_columns": config.state_columns,
                    "seed": config.seed,
                },
            )
            save_path = str(run_dir)

        best_prompt = result.best_candidate.get("system_prompt", "")
        display.set_result(best_prompt=best_prompt, save_path=save_path)

    return result
