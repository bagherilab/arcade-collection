from itertools import groupby

from prefect import task


@task
def group_template_conditions(conditions: list[dict], max_seeds: int) -> list[dict]:
    grouped_conditions = group_seed_ranges(conditions, max_seeds)
    condition_sets = group_condition_sets(grouped_conditions, max_seeds)
    template_conditions = [{"conditions": condition_set} for condition_set in condition_sets]
    return template_conditions


def group_seed_ranges(conditions: list[dict], max_seeds: int) -> list[dict]:
    conditions.sort(key=lambda x: (x["key"], x["seed"]))
    grouped_conditions = []

    for _, condition_group in groupby(conditions, lambda x: x["key"]):
        key_seeds = []
        key_conditions = {}

        for condition in condition_group:
            key_seeds.append(condition["seed"])
            key_conditions.update(condition)

        key_conditions.pop("seed")
        seed_ranges = find_seed_ranges(key_seeds, max_seeds)

        for range_start, range_end in seed_ranges:
            group_condition = key_conditions.copy()
            group_condition["start_seed"] = range_start
            group_condition["end_seed"] = range_end
            grouped_conditions.append(group_condition)

    return grouped_conditions


def find_seed_ranges(seeds: list[int], max_seeds: int) -> list[tuple[int, int]]:
    seeds.sort()
    ranges = []

    for _, group in groupby(enumerate(seeds), lambda x: x[0] - x[1]):
        group_list = list(group)
        subset = True
        range_start = group_list[0][1]
        range_end = group_list[-1][1]

        while subset:
            ranges.append((range_start, min(range_end, range_start + max_seeds - 1)))
            range_start = range_start + max_seeds
            if range_start > range_end:
                subset = False

    return ranges


def group_condition_sets(conditions: list[dict], max_seeds: int) -> list[list[dict]]:
    seed_count = 0
    condition_set = []
    condition_sets = []

    for condition in conditions:
        num_seeds = condition["end_seed"] - condition["start_seed"] + 1

        if seed_count + num_seeds <= max_seeds:
            condition_set.append(condition)
            seed_count = seed_count + num_seeds
        else:
            condition_sets.append(condition_set)
            seed_count = num_seeds
            condition_set = [condition]

    condition_sets.append(condition_set)

    return condition_sets
