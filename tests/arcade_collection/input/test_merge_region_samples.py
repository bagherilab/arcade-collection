import unittest

import pandas as pd

from arcade_collection.input.merge_region_samples import (
    filter_valid_samples,
    merge_region_samples,
    transform_sample_coordinates,
)


class TestMergeRegionSamples(unittest.TestCase):
    def test_merge_region_samples_no_regions(self):
        samples = {
            "DEFAULT": pd.DataFrame(
                {
                    "id": [1, 1, 1, 2, 2],
                    "x": [0, 1, 1, 2, 2],
                    "y": [3, 3, 4, 5, 5],
                    "z": [6, 6, 7, 7, 8],
                }
            ),
        }
        margins = (10, 20, 30)

        expected_merged = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [11, 12, 12, 13, 13],
                "y": [21, 21, 22, 23, 23],
                "z": [31, 31, 32, 32, 33],
            }
        )

        merged = merge_region_samples(samples, margins)

        self.assertTrue(expected_merged.equals(merged))

    def test_merge_region_samples_with_regions_no_fill(self):
        samples = {
            "DEFAULT": pd.DataFrame(
                {
                    "id": [1, 1, 1, 2, 2],
                    "x": [0, 1, 1, 2, 2],
                    "y": [3, 3, 4, 5, 5],
                    "z": [6, 6, 7, 7, 8],
                }
            ),
            "REGION_A": pd.DataFrame(
                {"id": [1, 1, 2], "x": [0, 1, 2], "y": [3, 4, 5], "z": [6, 7, 7]}
            ),
            "REGION_B": pd.DataFrame({"id": [1, 2], "x": [1, 2], "y": [3, 5], "z": [6, 8]}),
        }
        margins = (10, 20, 30)

        expected_merged = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [11, 12, 12, 13, 13],
                "y": [21, 21, 22, 23, 23],
                "z": [31, 31, 32, 32, 33],
                "region": ["REGION_A", "REGION_B", "REGION_A", "REGION_A", "REGION_B"],
            }
        )

        merged = merge_region_samples(samples, margins)

        self.assertTrue(expected_merged.equals(merged))

    def test_merge_region_samples_with_regions_with_fill(self):
        samples = {
            "DEFAULT": pd.DataFrame(
                {
                    "id": [1, 1, 1, 2, 2],
                    "x": [0, 1, 1, 2, 2],
                    "y": [3, 3, 4, 5, 5],
                    "z": [6, 6, 7, 7, 8],
                }
            ),
            "REGION_A": pd.DataFrame(
                {"id": [1, 1, 2], "x": [0, 1, 2], "y": [3, 4, 5], "z": [6, 7, 7]}
            ),
        }
        margins = (10, 20, 30)

        expected_merged = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [11, 12, 12, 13, 13],
                "y": [21, 21, 22, 23, 23],
                "z": [31, 31, 32, 32, 33],
                "region": ["REGION_A", "DEFAULT", "REGION_A", "REGION_A", "DEFAULT"],
            }
        )

        merged = merge_region_samples(samples, margins)

        self.assertTrue(expected_merged.equals(merged))

    def test_transform_sample_coordinates_no_reference(self):
        samples = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
            }
        )
        margins = (10, 20, 30)
        reference = None

        expected_transformed = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [11, 12, 12, 13, 13],
                "y": [21, 21, 22, 23, 23],
                "z": [31, 31, 32, 32, 33],
            }
        )

        transformed = transform_sample_coordinates(samples, margins, reference)

        self.assertTrue(expected_transformed.equals(transformed))

    def test_transform_sample_coordinates_with_reference(self):
        samples = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
            }
        )
        margins = (10, 20, 30)
        reference = pd.DataFrame(
            {
                "x": [0],
                "y": [1],
                "z": [2],
            }
        )

        expected_transformed = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [11, 12, 12, 13, 13],
                "y": [23, 23, 24, 25, 25],
                "z": [35, 35, 36, 36, 37],
            }
        )

        transformed = transform_sample_coordinates(samples, margins, reference)

        self.assertTrue(expected_transformed.equals(transformed))

    def test_filter_valid_samples_no_region_all_valid(self):
        samples = pd.DataFrame(
            {
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
            }
        )

        expected_filtered = samples.copy()

        filtered = filter_valid_samples(samples)

        self.assertTrue(expected_filtered.equals(filtered))

    def test_filter_valid_samples_with_region_all_valid(self):
        samples = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
                "region": ["A", "B", "B", "A", "B"],
            }
        )

        expected_filtered = samples.copy()

        filtered = filter_valid_samples(samples)

        self.assertTrue(expected_filtered.equals(filtered))

    def test_filter_valid_samples_sample_outside_region(self):
        samples = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "x": [0, 1, 1, 2, 2],
                "y": [3, 3, 4, 5, 5],
                "z": [6, 6, 7, 7, 8],
                "region": ["A", "B", "B", "A", "A"],
            }
        )

        expected_filtered = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "x": [0, 1, 1],
                "y": [3, 3, 4],
                "z": [6, 6, 7],
                "region": ["A", "B", "B"],
            }
        )

        filtered = filter_valid_samples(samples)

        self.assertTrue(expected_filtered.equals(filtered))


if __name__ == "__main__":
    unittest.main()
