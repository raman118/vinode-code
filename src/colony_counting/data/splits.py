"""Deterministic approximate stratified splitting with group separation."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import replace

from colony_counting.data.coco import ImageRecord


def split_by_group(
    records: list[ImageRecord],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> list[ImageRecord]:
    """Assign records to train/validation/test without splitting a group.

    Greedy allocation minimizes normalized squared deviations from target image counts and
    count-bin proportions. Groups are indivisible, so ratios and stratification are approximate.
    For datasets with fewer than three distinct groups, some partitions may be empty.
    """

    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values for train/val/test")
    ratio_sum = sum(ratios)
    ratios = tuple(ratio / ratio_sum for ratio in ratios)
    if not records:
        return []
    ids = [record.image_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("image_id values must be unique before splitting")

    split_groups = _split_group_ids(records)
    groups: dict[str, list[ImageRecord]] = {}
    for split_group_id, record in zip(split_groups, records, strict=True):
        groups.setdefault(split_group_id, []).append(record)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    bins = ("empty", "low", "medium", "high")
    group_vectors: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for group_id, members in group_items:
        counts = Counter(_count_bin(record.count) for record in members)
        group_vectors[group_id] = counts
        totals.update(counts)

    assigned: list[Counter[str]] = [Counter(), Counter(), Counter()]
    group_split: dict[str, str] = {}
    split_names = ("train", "val", "test")
    # Lowest incremental global allocation error wins. Seeded shuffle breaks exact ties.
    for group_id, _members in group_items:
        vector = group_vectors[group_id]
        candidate_costs: list[tuple[float, int]] = []
        for candidate in range(3):
            cost = 0.0
            for split_index in range(3):
                for stratum in bins:
                    target = totals[stratum] * ratios[split_index]
                    observed = assigned[split_index][stratum]
                    if split_index == candidate:
                        observed += vector[stratum]
                    cost += ((observed - target) ** 2) / max(target, 1.0)
            candidate_costs.append((cost, candidate))
        best_cost = min(cost for cost, _ in candidate_costs)
        tied = [index for cost, index in candidate_costs if abs(cost - best_cost) < 1e-12]
        chosen = rng.choice(tied)
        group_split[group_id] = split_names[chosen]
        assigned[chosen].update(vector)

    output = [
        replace(
            record,
            split=group_split[split_group_id],
            split_group_id=split_group_id,
        )
        for record, split_group_id in zip(records, split_groups, strict=True)
    ]
    validate_group_splits(output)
    return output


def validate_group_splits(records: list[ImageRecord]) -> None:
    """Raise when an image ID repeats or any group is assigned to multiple splits."""

    seen_ids: set[int | str] = set()
    group_splits: dict[str, set[str | None]] = {}
    for record in records:
        if record.image_id in seen_ids:
            raise ValueError(f"Duplicate image_id in split manifest: {record.image_id!r}")
        seen_ids.add(record.image_id)
        split_group = record.split_group_id or record.group_id
        group_splits.setdefault(split_group, set()).add(record.split)
    hash_splits: dict[str, set[str | None]] = {}
    for record in records:
        if record.sha256:
            hash_splits.setdefault(record.sha256, set()).add(record.split)
    leaked = sorted(group_id for group_id, values in group_splits.items() if len(values) != 1)
    leaked.extend(
        f"duplicate-sha256:{content_hash}"
        for content_hash, values in hash_splits.items()
        if len(values) != 1
    )
    if leaked:
        raise ValueError(f"Groups occur in multiple partitions: {leaked[:10]}")


def _count_bin(count: int) -> str:
    if count == 0:
        return "empty"
    if count <= 20:
        return "low"
    if count <= 100:
        return "medium"
    return "high"


def _split_group_ids(records: list[ImageRecord]) -> list[str]:
    """Join explicit groups and byte-identical images into indivisible split groups."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen_group: dict[str, int] = {}
    seen_hash: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.group_id in seen_group:
            union(index, seen_group[record.group_id])
        else:
            seen_group[record.group_id] = index
        if record.sha256:
            if record.sha256 in seen_hash:
                union(index, seen_hash[record.sha256])
            else:
                seen_hash[record.sha256] = index
    members: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        members.setdefault(find(index), []).append(str(record.image_id))
    labels = {
        root: "split-" + hashlib.sha256("\n".join(sorted(image_ids)).encode()).hexdigest()[:16]
        for root, image_ids in members.items()
    }
    return [labels[find(index)] for index in range(len(records))]
