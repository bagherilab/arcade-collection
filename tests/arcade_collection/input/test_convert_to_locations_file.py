import unittest

import pandas as pd

from arcade_collection.input.convert_to_locations_file import (
    convert_to_location,
    convert_to_locations_file,
    get_center_voxel,
    get_location_voxels,
)


class TestConvertToLocationsFile(unittest.TestCase):
    def test_convert_to_locations_file(self):
        samples = pd.DataFrame(
            {
                "id": [10, 10, 10, 10, 10, 11, 11, 11, 11],
                "x": [0, 1, 1, 2, 2, 30, 31, 31, 32],
                "y": [3, 3, 4, 5, 5, 40, 42, 42, 44],
                "z": [6, 6, 7, 7, 8, 50, 51, 52, 52],
                "region": [None, None, None, None, None, "A", "B", "A", "B"],
            }
        )

        expected_locations = [
            {
                "id": 1,
                "center": (1, 4, 6),
                "location": [
                    {
                        "region": "UNDEFINED",
                        "voxels": [
                            (0, 3, 6),
                            (1, 3, 6),
                            (1, 4, 7),
                            (2, 5, 7),
                            (2, 5, 8),
                        ],
                    }
                ],
            },
            {
                "id": 2,
                "center": (31, 42, 51),
                "location": [
                    {
                        "region": "A",
                        "voxels": [
                            (30, 40, 50),
                            (31, 42, 52),
                        ],
                    },
                    {
                        "region": "B",
                        "voxels": [
                            (31, 42, 51),
                            (32, 44, 52),
                        ],
                    },
                ],
            },
        ]

        locations = convert_to_locations_file(samples)

        self.assertCountEqual(expected_locations, locations)

    def test_convert_to_location_no_region(self):
        cell_id = 2
        samples = pd.DataFrame(
            {
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
            }
        )
        center = (1, 4, 6)

        expected_location = {
            "id": cell_id,
            "center": center,
            "location": [
                {
                    "region": "UNDEFINED",
                    "voxels": [
                        (0, 3, 6),
                        (1, 3, 6),
                        (1, 4, 7),
                        (2, 5, 7),
                        (2, 5, 8),
                    ],
                }
            ],
        }

        location = convert_to_location(cell_id, samples)

        self.assertDictEqual(expected_location, location)

    def test_convert_to_location_with_region(self):
        cell_id = 2
        samples = pd.DataFrame(
            {
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
                "region": ["A", "B", "A", "B", "A"],
            }
        )
        center = (1, 4, 6)

        expected_location = {
            "id": cell_id,
            "center": center,
            "location": [
                {
                    "region": "A",
                    "voxels": [
                        (0, 3, 6),
                        (1, 4, 7),
                        (2, 5, 8),
                    ],
                },
                {
                    "region": "B",
                    "voxels": [
                        (1, 3, 6),
                        (2, 5, 7),
                    ],
                },
            ],
        }

        location = convert_to_location(cell_id, samples)

        self.assertDictEqual(expected_location, location)

    def test_get_center_voxel(self):
        parameters = [
            ([10, 12], [3, 5], [2, 4], (11, 4, 3)),  # exact
            ([10, 11], [3, 4], [2, 3], (10, 3, 2)),  # rounded
        ]

        for x, y, z, expected_center in parameters:
            with self.subTest(x=x, y=y, z=z):
                samples = pd.DataFrame({"x": x, "y": y, "z": z})
                center = get_center_voxel(samples)
                self.assertTupleEqual(expected_center, center)

    def test_get_location_voxels_no_region(self):
        region = None
        samples = pd.DataFrame(
            {
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
            }
        )

        expected_voxels = [
            (0, 3, 6),
            (1, 3, 6),
            (1, 4, 7),
            (2, 5, 7),
            (2, 5, 8),
        ]

        voxels = get_location_voxels(samples, region)

        self.assertCountEqual(expected_voxels, voxels)

    def test_get_location_voxels_with_region(self):
        region = "A"
        samples = pd.DataFrame(
            {
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
                "region": ["A", "B", "A", "B", "A"],
            }
        )

        expected_voxels = [
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
        ]

        voxels = get_location_voxels(samples, region)

        self.assertCountEqual(expected_voxels, voxels)


if __name__ == "__main__":
    unittest.main()
