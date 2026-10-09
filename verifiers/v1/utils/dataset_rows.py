"""Bounded batch decoding for finite Hugging Face task datasets."""

from collections.abc import Iterable, Iterator
from typing import Any


def iter_dataset_rows(rows: Iterable[dict[str, Any]], batch_size: int = 256) -> Iterator[dict[str, Any]]:
    """Preserve row order and feature decoding without formatting each Arrow slice."""
    from datasets import Dataset

    if not isinstance(rows, Dataset):
        yield from rows
        return
    for batch in rows.iter(batch_size=batch_size):
        keys = tuple(batch)
        yield from (dict(zip(keys, values, strict=True)) for values in zip(*(batch[key] for key in keys), strict=True))
