"""Reproducible task sampling shared by the run entrypoints.

`eval`, `validate`, `debug`, the v0 bridge, and `gepa` all select a run's tasks the same way:
optionally shuffle under a fixed seed (so a `--shuffle` run samples the same tasks every time),
then take the first `num`. Generic over the item type, so it works on `Task` lists and bare
task-index lists alike.
"""

import random
from typing import TypeVar

T = TypeVar("T")

_SHUFFLE_SEED = (
    0  # fixed so `--shuffle` samples the same tasks every run (reproducible)
)


def sample_tasks(
    items: list[T], num: int | None = None, shuffle: bool = False
) -> list[T]:
    """Up to `num` of `items` (all if `None`), optionally shuffled under the fixed seed. Copies
    the input rather than shuffling in place."""
    items = list(items)
    if shuffle:
        random.Random(_SHUFFLE_SEED).shuffle(items)
    return items if num is None else items[:num]
