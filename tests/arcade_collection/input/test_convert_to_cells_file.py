from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from arcade_collection.input.convert_to_cells_file import (
    convert_to_cell,
    convert_to_cell_region,
    convert_to_cells_file,
    convert_value_distribution,
    filter_cell_reference,
    get_cell_state,
)

EPSILON = 1e-10
DEFAULT_REGION_NAME = "DEFAULT"


def make_samples(cell_id: int, volume: int, height: int, region: str | None):
    return pd.DataFrame(
        {
            "id": [cell_id] * volume,
            "z": np.linspace(0, height, volume),
            "region": [region] * volume,
        }
    )


class TestConvertToCellsFile(unittest.TestCase):
    def setUp(self):
        self.volume_distributions = {
            DEFAULT_REGION_NAME: (10, 10),
            "REGION1": (10, 5),
            "REGION2": (25, 10),
        }
        self.height_distributions = {
            DEFAULT_REGION_NAME: (5, 5),
            "REGION1": (4, 1),
            "REGION2": (10, 5),
        }
        self.critical_volume_distributions = {
            DEFAULT_REGION_NAME: (2, 2),
            "REGION1": (2, 4),
            "REGION2": (6, 1),
        }
        self.critical_height_distributions = {
            DEFAULT_REGION_NAME: (10, 10),
            "REGION1": (2, 3),
            "REGION2": (8, 4),
        }
        self.state_thresholds = {
            "STATE1_PHASE1": 0.2,
            "STATE1_PHASE2": 1.5,
            "STATE2_PHASE1": 2,
        }

        self.reference = {
            "volume": 10,
            "height": 10,
            "volume.REGION1": 10,
            "height.REGION1": 4,
            "volume.REGION2": 15,
            "height.REGION2": 10,
        }

    def test_convert_to_cells_file(self):
        cell_ids = [10, 11, 12, 13]
        volumes = [[20], [20], [15, 5], [40, 10]]
        heights = [5, 5, 5, 20]

        samples = pd.concat(
            [
                (
                    make_samples(cell_id, volume, height, f"REGION{index + 1}")
                    if len(region_volumes) > 1
                    else make_samples(cell_id, volume, height, None)
                )
                for cell_id, height, region_volumes in zip(cell_ids, heights, volumes)
                for index, volume in enumerate(region_volumes)
            ]
        )

        reference = pd.DataFrame(
            {
                "ID": [11, 13],
                "volume": [10, 10],
                "height": [10, 10],
                "volume.REGION1": [None, 10],
                "height.REGION1": [None, 4],
                "volume.REGION2": [None, 15],
                "height.REGION2": [None, 10],
            }
        )

        expected_cells = [
            {
                "id": 1,
                "parent": 0,
                "pop": 1,
                "age": 0,
                "divisions": 0,
                "state": "STATE2",
                "phase": "STATE2_PHASE1",
                "voxels": np.sum(volumes[0]),
                "criticals": [4, 10],
            },
            {
                "id": 2,
                "parent": 0,
                "pop": 1,
                "age": 0,
                "divisions": 0,
                "state": "STATE2",
                "phase": "STATE2_PHASE1",
                "voxels": np.sum(volumes[1]),
                "criticals": [2, 20],
            },
            {
                "id": 3,
                "parent": 0,
                "pop": 1,
                "age": 0,
                "divisions": 0,
                "state": "STATE2",
                "phase": "STATE2_PHASE1",
                "voxels": np.sum(volumes[2]),
                "criticals": [4, 10],
                "regions": [
                    {"region": "REGION1", "voxels": volumes[2][0], "criticals": [6, 5]},
                    {"region": "REGION2", "voxels": volumes[2][1], "criticals": [4, 4]},
                ],
            },
            {
                "id": 4,
                "parent": 0,
                "pop": 1,
                "age": 0,
                "divisions": 0,
                "state": "STATE2",
                "phase": "STATE2_PHASE1",
                "voxels": np.sum(volumes[3]),
                "criticals": [2, 20],
                "regions": [
                    {"region": "REGION1", "voxels": volumes[3][0], "criticals": [2, 2]},
                    {"region": "REGION2", "voxels": volumes[3][1], "criticals": [5, 8]},
                ],
            },
        ]

        cells = convert_to_cells_file(
            samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
            self.state_thresholds,
        )

        self.assertCountEqual(expected_cells, cells)

    def test_convert_to_cell_no_reference_no_region(self):
        cell_id = 2
        volume = 20
        height = 5
        samples = make_samples(cell_id, volume, height, None)
        reference = {}

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE2",
            "phase": "STATE2_PHASE1",
            "voxels": volume,
            "criticals": [4, 10],
        }

        cell = convert_to_cell(
            cell_id,
            samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
            self.state_thresholds,
        )

        self.assertDictEqual(expected_cell, cell)

    def test_convert_to_cell_with_reference_no_region(self):
        cell_id = 2
        volume = 20
        height = 5
        samples = make_samples(cell_id, volume, height, None)
        reference = {"volume": 10, "height": 10}

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE2",
            "phase": "STATE2_PHASE1",
            "voxels": volume,
            "criticals": [2, 20],
        }

        cell = convert_to_cell(
            cell_id,
            samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
            self.state_thresholds,
        )

        self.assertDictEqual(expected_cell, cell)

    def test_convert_to_cell_no_reference_with_region(self):
        cell_id = 2
        volumes = [15, 5]
        height = 5
        samples = pd.concat(
            [
                make_samples(cell_id, volume, height, f"REGION{index + 1}")
                for index, volume in enumerate(volumes)
            ]
        )
        reference = {}

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE2",
            "phase": "STATE2_PHASE1",
            "voxels": np.sum(volumes),
            "criticals": [4, 10],
            "regions": [
                {"region": "REGION1", "voxels": volumes[0], "criticals": [6, 5]},
                {"region": "REGION2", "voxels": volumes[1], "criticals": [4, 4]},
            ],
        }

        cell = convert_to_cell(
            cell_id,
            samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
            self.state_thresholds,
        )

        self.assertDictEqual(expected_cell, cell)

    def test_convert_to_cell_with_reference_with_region(self):
        cell_id = 2
        volumes = [15, 5]
        height = 5
        samples = pd.concat(
            [
                make_samples(cell_id, volume, height, f"REGION{index + 1}")
                for index, volume in enumerate(volumes)
            ]
        )
        reference = {
            "volume": 10,
            "height": 10,
            "volume.REGION1": 10,
            "height.REGION1": 4,
            "volume.REGION2": 15,
            "height.REGION2": 10,
        }

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE2",
            "phase": "STATE2_PHASE1",
            "voxels": np.sum(volumes),
            "criticals": [2, 20],
            "regions": [
                {"region": "REGION1", "voxels": volumes[0], "criticals": [2, 2]},
                {"region": "REGION2", "voxels": volumes[1], "criticals": [5, 8]},
            ],
        }

        cell = convert_to_cell(
            cell_id,
            samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
            self.state_thresholds,
        )

        self.assertDictEqual(expected_cell, cell)

    def test_convert_to_cell_region_no_reference(self):
        volume = 20
        height = 5
        region_samples = pd.DataFrame({"z": np.linspace(0, height, volume)})
        reference = {}

        expected_cell_region = {
            "region": DEFAULT_REGION_NAME,
            "voxels": volume,
            "criticals": [4, 10],
        }

        cell_region = convert_to_cell_region(
            DEFAULT_REGION_NAME,
            region_samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
        )

        self.assertDictEqual(expected_cell_region, cell_region)

    def test_convert_to_cell_region_with_reference(self):
        volume = 20
        height = 5
        region_samples = pd.DataFrame({"z": np.linspace(0, height, volume)})
        reference = {f"volume.{DEFAULT_REGION_NAME}": 10, f"height.{DEFAULT_REGION_NAME}": 10}

        expected_cell_region = {
            "region": DEFAULT_REGION_NAME,
            "voxels": volume,
            "criticals": [2, 20],
        }

        cell_region = convert_to_cell_region(
            DEFAULT_REGION_NAME,
            region_samples,
            reference,
            self.volume_distributions,
            self.height_distributions,
            self.critical_volume_distributions,
            self.critical_height_distributions,
        )

        self.assertDictEqual(expected_cell_region, cell_region)

    def test_get_cell_state(self):
        critical_volume = 10
        threshold_fractions = {
            "STATE_A": 0.2,
            "STATE_B": 1.5,
            "STATE_C": 2,
        }

        threshold_a = threshold_fractions["STATE_A"] * critical_volume
        threshold_b = threshold_fractions["STATE_B"] * critical_volume
        threshold_c = threshold_fractions["STATE_C"] * critical_volume

        parameters = [
            (threshold_a - EPSILON, "STATE_A"),  # below A threshold
            (threshold_a, "STATE_B"),  # equal A threshold
            ((threshold_a + threshold_b) / 2, "STATE_B"),  # between A and B thresholds
            (threshold_b, "STATE_C"),  # equal B threshold
            ((threshold_b + threshold_c) / 2, "STATE_C"),  # between B and C thresholds
            (threshold_c, "STATE_C"),  # equal C threshold
            (threshold_c + EPSILON, "STATE_C"),  # above C threshold
        ]

        for volume, expected_phase in parameters:
            with self.subTest(volume=volume, expected_phase=expected_phase):
                phase = get_cell_state(volume, critical_volume, threshold_fractions)
                self.assertEqual(expected_phase, phase)

    def test_convert_value_distribution(self):
        source_distribution = (10, 6)
        target_distribution = (2, 0.6)

        parameters = [
            (10, 2),  # means
            (4, 1.4),  # one standard deviation below
            (16, 2.6),  # one standard deviation above
            (7, 1.7),  # half standard deviation below
            (13, 2.3),  # half standard deviation above
        ]

        for source_value, expected_target_value in parameters:
            with self.subTest(source_value=source_value):
                target_value = convert_value_distribution(
                    source_value, source_distribution, target_distribution
                )
                self.assertEqual(expected_target_value, target_value)

    def test_filter_cell_reference_cell_exists(self):
        cell_ids = [1, 2, 3]
        feature_a = [10, 20, 30]
        feature_b = ["a", "b", "c"]
        cell_id = 2
        index = cell_ids.index(cell_id)

        reference = pd.DataFrame({"ID": cell_ids, "FEATURE_A": feature_a, "FEATURE_B": feature_b})

        expected_cell_reference = {
            "ID": cell_id,
            "FEATURE_A": feature_a[index],
            "FEATURE_B": feature_b[index],
        }

        cell_reference = filter_cell_reference(cell_id, reference)

        self.assertDictEqual(expected_cell_reference, cell_reference)

    def test_filter_cell_reference_cell_does_not_exist(self):
        cell_ids = [1, 2, 3]
        feature_a = [10, 20, 30]
        feature_b = ["a", "b", "c"]
        cell_id = 4

        reference = pd.DataFrame({"ID": cell_ids, "FEATURE_A": feature_a, "FEATURE_B": feature_b})

        cell_reference = filter_cell_reference(cell_id, reference)

        self.assertDictEqual({}, cell_reference)


if __name__ == "__main__":
    unittest.main()
