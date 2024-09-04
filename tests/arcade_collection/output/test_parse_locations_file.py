import json
import tarfile
import unittest
from unittest import mock

import pandas as pd

from arcade_collection.output.parse_locations_file import parse_location_tick, parse_locations_file


class TestParseLocationsFile(unittest.TestCase):
    def test_parse_locations_file_without_regions(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        first_member_mock = mock.Mock(spec=tarfile.ExFileObject)
        second_member_mock = mock.Mock(spec=tarfile.ExFileObject)

        first_member_mock.name = "key_000005.LOCATIONS.json"
        second_member_mock.name = "key_000006.LOCATIONS.json"

        contents = {
            first_member_mock.name: first_member_mock,
            second_member_mock.name: second_member_mock,
        }

        tar_mock.getmembers.return_value = [*list(contents.values()), None]
        tar_mock.extractfile.side_effect = lambda m: None if m is None else contents[m.name]

        first_member_contents = [
            {
                "id": 1,
                "center": [2, 3, 4],
                "location": [
                    {
                        "region": "UNDEFINED",
                        "voxels": [[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]],
                    },
                ],
            },
            {
                "id": 17,
                "center": [18, 19, 20],
                "location": [
                    {
                        "region": "UNDEFINED",
                        "voxels": [[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32]],
                    },
                ],
            },
        ]
        second_member_contents = [
            {
                "id": 33,
                "center": [34, 35, 36],
                "location": [
                    {
                        "region": "UNDEFINED",
                        "voxels": [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
                    },
                ],
            }
        ]

        first_member_mock.read.return_value = json.dumps(first_member_contents).encode("utf-8")
        second_member_mock.read.return_value = json.dumps(second_member_contents).encode("utf-8")

        regions = []

        expected_data = {
            "ID": [1, 17, 33],
            "TICK": [5, 5, 6],
            "CENTER_X": [2, 18, 34],
            "CENTER_Y": [3, 19, 35],
            "CENTER_Z": [4, 20, 36],
            "MIN_X": [5, 21, 37],
            "MIN_Y": [6, 22, 38],
            "MIN_Z": [7, 23, 39],
            "MAX_X": [14, 30, 46],
            "MAX_Y": [15, 31, 47],
            "MAX_Z": [16, 32, 48],
        }

        data = parse_locations_file(tar_mock, regions)

        self.assertTrue(pd.DataFrame(expected_data).equals(data))

    def test_parse_locations_file_with_regions(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        first_member_mock = mock.Mock(spec=tarfile.ExFileObject)
        second_member_mock = mock.Mock(spec=tarfile.ExFileObject)

        first_member_mock.name = "key_000005.LOCATIONS.json"
        second_member_mock.name = "key_000006.LOCATIONS.json"

        contents = {
            first_member_mock.name: first_member_mock,
            second_member_mock.name: second_member_mock,
        }

        tar_mock.getmembers.return_value = [*list(contents.values()), None]
        tar_mock.extractfile.side_effect = lambda m: None if m is None else contents[m.name]

        first_member_contents = [
            {
                "id": 1,
                "center": [2, 3, 4],
                "location": [
                    {"region": "REGION_A", "voxels": [[5, 6, 7], [8, 9, 10]]},
                    {"region": "REGION_B", "voxels": [[11, 12, 13], [14, 15, 16]]},
                ],
            },
            {
                "id": 17,
                "center": [18, 19, 20],
                "location": [
                    {"region": "REGION_A", "voxels": [[21, 22, 23], [24, 25, 26]]},
                    {"region": "REGION_B", "voxels": [[27, 28, 29], [30, 31, 32]]},
                ],
            },
        ]
        second_member_contents = [
            {
                "id": 33,
                "center": [34, 35, 36],
                "location": [
                    {
                        "region": "REGION_A",
                        "voxels": [[37, 38, 39], [40, 41, 42]],
                    },
                    {
                        "region": "REGION_B",
                        "voxels": [[43, 44, 45], [46, 47, 48]],
                    },
                ],
            }
        ]

        first_member_mock.read.return_value = json.dumps(first_member_contents).encode("utf-8")
        second_member_mock.read.return_value = json.dumps(second_member_contents).encode("utf-8")

        regions = ["REGION_A", "REGION_B"]

        expected_data = {
            "ID": [1, 17, 33],
            "TICK": [5, 5, 6],
            "CENTER_X": [2, 18, 34],
            "CENTER_Y": [3, 19, 35],
            "CENTER_Z": [4, 20, 36],
            "MIN_X": [5, 21, 37],
            "MIN_Y": [6, 22, 38],
            "MIN_Z": [7, 23, 39],
            "MAX_X": [14, 30, 46],
            "MAX_Y": [15, 31, 47],
            "MAX_Z": [16, 32, 48],
            "CENTER_X.REGION_A": [7, 23, 39],
            "CENTER_Y.REGION_A": [8, 24, 40],
            "CENTER_Z.REGION_A": [9, 25, 41],
            "MIN_X.REGION_A": [5, 21, 37],
            "MIN_Y.REGION_A": [6, 22, 38],
            "MIN_Z.REGION_A": [7, 23, 39],
            "MAX_X.REGION_A": [8, 24, 40],
            "MAX_Y.REGION_A": [9, 25, 41],
            "MAX_Z.REGION_A": [10, 26, 42],
            "CENTER_X.REGION_B": [13, 29, 45],
            "CENTER_Y.REGION_B": [14, 30, 46],
            "CENTER_Z.REGION_B": [15, 31, 47],
            "MIN_X.REGION_B": [11, 27, 43],
            "MIN_Y.REGION_B": [12, 28, 44],
            "MIN_Z.REGION_B": [13, 29, 45],
            "MAX_X.REGION_B": [14, 30, 46],
            "MAX_Y.REGION_B": [15, 31, 47],
            "MAX_Z.REGION_B": [16, 32, 48],
        }

        data = parse_locations_file(tar_mock, regions)

        self.assertTrue(pd.DataFrame(expected_data).equals(data))

    def test_parse_location_tick_without_regions_empty_list(self):
        tick = 15
        regions = []
        location = {"id": 1, "location": [{"region": "UNDEFINED", "voxels": []}]}

        expected = [1, tick, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        parsed = parse_location_tick(tick, location, regions)

        self.assertListEqual(expected, parsed)

    def test_parse_location_tick_without_regions(self):
        tick = 15
        regions = []
        location = {
            "id": 1,
            "center": [2, 3, 4],
            "location": [
                {
                    "region": "UNDEFINED",
                    "voxels": [[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]],
                }
            ],
        }

        expected = [1, tick, 2, 3, 4, 5, 6, 7, 14, 15, 16]

        parsed = parse_location_tick(tick, location, regions)

        self.assertListEqual(expected, parsed)

    def test_parse_location_tick_with_regions_empty_list(self):
        tick = 15
        regions = ["REGION_A", "REGION_B"]
        location = {
            "id": 1,
            "center": [2, 3, 4],
            "location": [
                {"region": "REGION_A", "voxels": [[5, 6, 7], [8, 9, 10]]},
                {"region": "REGION_B", "voxels": []},
            ],
        }

        expected = [1, tick, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = [*expected, 7, 8, 9, 5, 6, 7, 8, 9, 10]  # REGION_A
        expected = [*expected, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # REGION_B

        parsed = parse_location_tick(tick, location, regions)

        self.assertListEqual(expected, parsed)

    def test_parse_location_tick_with_regions(self):
        tick = 15
        regions = ["REGION_A", "REGION_B"]
        location = {
            "id": 1,
            "center": [2, 3, 4],
            "location": [
                {"region": "REGION_A", "voxels": [[5, 6, 7], [8, 9, 10]]},
                {"region": "REGION_B", "voxels": [[11, 12, 13], [14, 15, 16]]},
            ],
        }

        expected = [1, tick, 2, 3, 4, 5, 6, 7, 14, 15, 16]
        expected = [*expected, 7, 8, 9, 5, 6, 7, 8, 9, 10]  # REGION_A
        expected = [*expected, 13, 14, 15, 11, 12, 13, 14, 15, 16]  # REGION_B

        parsed = parse_location_tick(tick, location, regions)

        self.assertListEqual(expected, parsed)


if __name__ == "__main__":
    unittest.main()
