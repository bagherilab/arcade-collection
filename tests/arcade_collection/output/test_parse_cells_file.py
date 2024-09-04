import json
import tarfile
import unittest
from unittest import mock

import pandas as pd

from arcade_collection.output.parse_cells_file import parse_cell_tick, parse_cells_file


class TestParseCellsFile(unittest.TestCase):
    def test_parse_cells_file_without_regions(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        first_member_mock = mock.Mock(spec=tarfile.ExFileObject)
        second_member_mock = mock.Mock(spec=tarfile.ExFileObject)

        first_member_mock.name = "key_000005.CELLS.json"
        second_member_mock.name = "key_000006.CELLS.json"

        contents = {
            first_member_mock.name: first_member_mock,
            second_member_mock.name: second_member_mock,
        }

        tar_mock.getmembers.return_value = contents.values()
        tar_mock.extractfile.side_effect = lambda member: contents.get(member.name, None)

        first_member_contents = [
            {
                "id": 1,
                "parent": 2,
                "pop": 3,
                "age": 4,
                "divisions": 5,
                "state": "STATE_A",
                "phase": "PHASE_A",
                "voxels": 6,
                "criticals": [7, 8],
            },
            {
                "id": 15,
                "parent": 16,
                "pop": 17,
                "age": 18,
                "divisions": 19,
                "state": "STATE_B",
                "phase": "PHASE_B",
                "voxels": 20,
                "criticals": [21, 22],
            },
        ]
        second_member_contents = [
            {
                "id": 29,
                "parent": 30,
                "pop": 31,
                "age": 32,
                "divisions": 33,
                "state": "STATE_C",
                "phase": "PHASE_C",
                "voxels": 34,
                "criticals": [35, 36],
            }
        ]

        first_member_mock.read.return_value = json.dumps(first_member_contents).encode("utf-8")
        second_member_mock.read.return_value = json.dumps(second_member_contents).encode("utf-8")

        regions = []

        expected_data = {
            "ID": [1, 15, 29],
            "TICK": [5, 5, 6],
            "PARENT": [2, 16, 30],
            "POPULATION": [3, 17, 31],
            "AGE": [4, 18, 32],
            "DIVISIONS": [5, 19, 33],
            "STATE": ["STATE_A", "STATE_B", "STATE_C"],
            "PHASE": ["PHASE_A", "PHASE_B", "PHASE_C"],
            "NUM_VOXELS": [6, 20, 34],
        }

        data = parse_cells_file(tar_mock, regions)

        self.assertTrue(pd.DataFrame(expected_data).equals(data))

    def test_parse_cells_file_with_regions(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        first_member_mock = mock.Mock(spec=tarfile.ExFileObject)
        second_member_mock = mock.Mock(spec=tarfile.ExFileObject)

        first_member_mock.name = "key_000005.CELLS.json"
        second_member_mock.name = "key_000006.CELLS.json"

        contents = {
            first_member_mock.name: first_member_mock,
            second_member_mock.name: second_member_mock,
        }

        tar_mock.getmembers.return_value = contents.values()
        tar_mock.extractfile.side_effect = lambda member: contents.get(member.name, None)

        first_member_contents = [
            {
                "id": 1,
                "parent": 2,
                "pop": 3,
                "age": 4,
                "divisions": 5,
                "state": "STATE_A",
                "phase": "PHASE_A",
                "voxels": 6,
                "criticals": [7, 8],
                "regions": [
                    {"region": "REGION_A", "voxels": 9, "criticals": [10, 11]},
                    {"region": "REGION_B", "voxels": 12, "criticals": [13, 14]},
                ],
            },
            {
                "id": 15,
                "parent": 16,
                "pop": 17,
                "age": 18,
                "divisions": 19,
                "state": "STATE_B",
                "phase": "PHASE_B",
                "voxels": 20,
                "criticals": [21, 22],
                "regions": [
                    {"region": "REGION_A", "voxels": 23, "criticals": [24, 25]},
                    {"region": "REGION_B", "voxels": 26, "criticals": [27, 28]},
                ],
            },
        ]
        second_member_contents = [
            {
                "id": 29,
                "parent": 30,
                "pop": 31,
                "age": 32,
                "divisions": 33,
                "state": "STATE_C",
                "phase": "PHASE_C",
                "voxels": 34,
                "criticals": [35, 36],
                "regions": [
                    {"region": "REGION_A", "voxels": 37, "criticals": [38, 39]},
                    {"region": "REGION_B", "voxels": 40, "criticals": [41, 42]},
                ],
            }
        ]

        first_member_mock.read.return_value = json.dumps(first_member_contents).encode("utf-8")
        second_member_mock.read.return_value = json.dumps(second_member_contents).encode("utf-8")

        regions = ["REGION_A", "REGION_B"]

        expected_data = {
            "ID": [1, 15, 29],
            "TICK": [5, 5, 6],
            "PARENT": [2, 16, 30],
            "POPULATION": [3, 17, 31],
            "AGE": [4, 18, 32],
            "DIVISIONS": [5, 19, 33],
            "STATE": ["STATE_A", "STATE_B", "STATE_C"],
            "PHASE": ["PHASE_A", "PHASE_B", "PHASE_C"],
            "NUM_VOXELS": [6, 20, 34],
            "NUM_VOXELS.REGION_A": [9, 23, 37],
            "NUM_VOXELS.REGION_B": [12, 26, 40],
        }

        data = parse_cells_file(tar_mock, regions)

        self.assertTrue(pd.DataFrame(expected_data).equals(data))

    def test_parse_cell_tick_without_regions(self):
        tick = 15
        regions = []
        cell = {
            "id": 1,
            "parent": 2,
            "pop": 3,
            "age": 4,
            "divisions": 5,
            "state": "STATE",
            "phase": "PHASE",
            "voxels": 6,
            "criticals": [7, 8],
        }

        expected = [1, tick, 2, 3, 4, 5, "STATE", "PHASE", 6]

        parsed = parse_cell_tick(tick, cell, regions)

        self.assertListEqual(expected, parsed)

    def test_parse_cell_tick_with_regions(self):
        tick = 15
        regions = ["REGION_A", "REGION_B"]
        cell = {
            "id": 1,
            "parent": 2,
            "pop": 3,
            "age": 4,
            "divisions": 5,
            "state": "STATE",
            "phase": "PHASE",
            "voxels": 6,
            "criticals": [7, 8],
            "regions": [
                {"region": "REGION_A", "voxels": 9, "criticals": [10, 11]},
                {"region": "REGION_B", "voxels": 12, "criticals": [13, 14]},
            ],
        }

        expected = [1, tick, 2, 3, 4, 5, "STATE", "PHASE", 6]
        expected = [*expected, 9]  # REGION_A
        expected = [*expected, 12]  # REGION_B

        parsed = parse_cell_tick(tick, cell, regions)

        self.assertListEqual(expected, parsed)


if __name__ == "__main__":
    unittest.main()
