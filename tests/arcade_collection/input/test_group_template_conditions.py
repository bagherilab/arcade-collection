import unittest

from arcade_collection.input.group_template_conditions import (
    find_seed_ranges,
    group_condition_sets,
    group_seed_ranges,
    group_template_conditions,
)


class TestGroupTemplateConditions(unittest.TestCase):
    def test_group_template_conditions(self):
        max_seeds = 3
        conditions = [
            {"key": "A", "seed": 1},
            {"key": "A", "seed": 2},
            {"key": "A", "seed": 4},
            {"key": "B", "seed": 1},
            {"key": "B", "seed": 2},
            {"key": "B", "seed": 3},
            {"key": "B", "seed": 4},
        ]

        expected_groups = [
            {
                "conditions": [
                    {"key": "A", "start_seed": 1, "end_seed": 2},
                    {"key": "A", "start_seed": 4, "end_seed": 4},
                ]
            },
            {
                "conditions": [
                    {"key": "B", "start_seed": 1, "end_seed": 3},
                ]
            },
            {
                "conditions": [
                    {"key": "B", "start_seed": 4, "end_seed": 4},
                ]
            },
        ]

        groups = group_template_conditions(conditions, max_seeds)

        self.assertCountEqual(expected_groups, groups)

    def test_find_seed_ranges_continuous(self):
        seeds = [0, 1, 2, 3]
        parameters = [
            (2, [(0, 1), (2, 3)]),  # below max, equal
            (3, [(0, 2), (3, 3)]),  # below max, unequal
            (4, [(0, 3)]),  # equal to max
            (5, [(0, 3)]),  # above max
        ]

        for max_seeds, expected_groups in parameters:
            with self.subTest(max_seeds=max_seeds):
                groups = find_seed_ranges(seeds, max_seeds)
                self.assertCountEqual(expected_groups, groups)

    def test_find_seed_ranges_discontinuous(self):
        seeds = [0, 1, 2, 3, 5, 6, 7, 8]
        parameters = [
            (2, [(0, 1), (2, 3), (5, 6), (7, 8)]),  # below max, equal
            (3, [(0, 2), (3, 3), (5, 7), (8, 8)]),  # below max, unequal
            (4, [(0, 3), (5, 8)]),  # equal to max
            (5, [(0, 3), (5, 8)]),  # above max
        ]

        for max_seeds, expected_groups in parameters:
            with self.subTest(max_seeds=max_seeds):
                groups = find_seed_ranges(seeds, max_seeds)
                self.assertCountEqual(expected_groups, groups)

    def test_group_seed_ranges(self):
        max_seeds = 2
        conditions = [
            {"key": "A", "seed": 1},
            {"key": "A", "seed": 2},
            {"key": "A", "seed": 3},
            {"key": "B", "seed": 1},
            {"key": "B", "seed": 2},
            {"key": "B", "seed": 3},
            {"key": "B", "seed": 4},
        ]

        expected_groups = [
            {"key": "A", "start_seed": 1, "end_seed": 2},
            {"key": "A", "start_seed": 3, "end_seed": 3},
            {"key": "B", "start_seed": 1, "end_seed": 2},
            {"key": "B", "start_seed": 3, "end_seed": 4},
        ]

        groups = group_seed_ranges(conditions, max_seeds)

        self.assertCountEqual(expected_groups, groups)

    def test_group_condition_sets(self):
        max_seeds = 3
        conditions = [
            {"key": "A", "start_seed": 1, "end_seed": 2},
            {"key": "A", "start_seed": 3, "end_seed": 3},
            {"key": "B", "start_seed": 1, "end_seed": 2},
            {"key": "B", "start_seed": 3, "end_seed": 4},
        ]

        expected_groups = [
            [
                {"key": "A", "start_seed": 1, "end_seed": 2},
                {"key": "A", "start_seed": 3, "end_seed": 3},
            ],
            [
                {"key": "B", "start_seed": 1, "end_seed": 2},
            ],
            [
                {"key": "B", "start_seed": 3, "end_seed": 4},
            ],
        ]

        groups = group_condition_sets(conditions, max_seeds)

        self.assertCountEqual(expected_groups, groups)


if __name__ == "__main__":
    unittest.main()
