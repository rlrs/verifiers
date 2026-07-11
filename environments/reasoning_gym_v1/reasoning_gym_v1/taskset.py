from typing import Any

import verifiers.v1 as vf
from pydantic import Field
from reasoning_gym import create_dataset

SYSTEM_PROMPT = (
    "Solve the problem carefully. Return the requested answer in the exact format implied "
    "by the examples or question. You may explain your reasoning before the final answer."
)


class ReasoningGymData(vf.TaskData):
    gym: str
    entry: dict[str, Any]
    dataset_config: dict[str, Any]
    seed: int
    size: int


class ReasoningGymTask(vf.Task[ReasoningGymData]):
    def _dataset(self):
        return create_dataset(
            self.data.gym,
            seed=self.data.seed,
            size=self.data.size,
            **self.data.dataset_config,
        )

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return float(self._dataset().score_answer(trace.last_reply, self.data.entry))

    async def validate(self, runtime: vf.Runtime) -> bool:
        del runtime
        answer = self.data.entry.get("answer")
        return float(self._dataset().score_answer(answer, self.data.entry)) == 1.0


class ReasoningGymConfig(vf.TasksetConfig):
    gym: str = "arc_1d"
    size: int = Field(2000, ge=1)
    seed: int = Field(0, ge=0)
    dataset_config: dict[str, Any] = Field(default_factory=dict)


class ReasoningGymTaskset(vf.Taskset[ReasoningGymTask, ReasoningGymConfig]):
    def load(self) -> list[ReasoningGymTask]:
        dataset = create_dataset(
            self.config.gym,
            seed=self.config.seed,
            size=self.config.size,
            **self.config.dataset_config,
        )
        return [
            ReasoningGymTask(
                ReasoningGymData(
                    idx=index,
                    name=f"{self.config.gym}_{index}",
                    prompt=f"{SYSTEM_PROMPT}\n\n{entry['question']}",
                    gym=self.config.gym,
                    entry=entry,
                    dataset_config=self.config.dataset_config,
                    seed=self.config.seed,
                    size=self.config.size,
                ),
                self.config.task,
            )
            for index, entry in enumerate(dataset)
        ]
