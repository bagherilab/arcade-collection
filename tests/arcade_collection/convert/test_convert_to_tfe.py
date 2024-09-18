import unittest

import pandas as pd

from arcade_collection.convert.convert_to_tfe import convert_to_tfe


class TestConvertToTFE(unittest.TestCase):
    def test_convert_to_tfe(self):
        all_data = pd.DataFrame(
            {
                "TICK": [0, 0, 0, 0, 5, 5, 5, 10, 10, 10, 15, 15],
                "ID": [1, 2, 4, 5, 1, 4, 5, 1, 2, 4, 1, 2],
                "time": [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5],
                "feature_a": [10, 12, 14, 16, 10, 12, 15, 11, 12, 15, 16, 10],
                "feature_b": ["A", "B", "A", "B", "A", "B", "C", "C", "C", "A", "B", "C"],
                "feature_c": [1.0, 1.2, 1.4, 1.6, 1.0, 1.2, 1.5, 1.1, 1.2, 1.5, 1.6, 1.0],
            }
        )
        features = [
            ("feature_a", "Feature A name", "continuous"),
            ("feature_b", "Feature B name", "categorical"),
            ("feature_c", "Feature C name", "discrete"),
        ]
        frame_spec = (0, 16, 10)

        expected_manifest = {
            "frames": ["frames/frame_0.png", "frames/frame_1.png"],
            "features": [
                {
                    "key": "feature_a",
                    "name": "Feature A name",
                    "data": "features/feature_a.json",
                    "type": "continuous",
                },
                {
                    "key": "feature_b",
                    "name": "Feature B name",
                    "data": "features/feature_b.json",
                    "type": "categorical",
                    "categories": ["A", "B", "C"],
                },
                {
                    "key": "feature_c",
                    "name": "Feature C name",
                    "data": "features/feature_c.json",
                    "type": "discrete",
                },
            ],
            "tracks": "tracks.json",
            "times": "times.json",
        }

        expected_tracks = {"data": [0, 1, 2, 4, 5, 1, 2, 4]}
        expected_times = {"data": [0, 0, 0, 0, 0, 1, 1, 1]}

        expected_feature_a = {"data": [0, 10, 12, 14, 16, 11, 12, 15], "min": 10, "max": 16}
        expected_feature_b = {"data": [0, 0, 1, 0, 1, 2, 2, 0], "min": 0, "max": 2}
        expected_feature_c = {"data": [0, 1, 1.2, 1.4, 1.6, 1.1, 1.2, 1.5], "min": 1, "max": 1.6}

        tfe = convert_to_tfe(all_data, features, frame_spec)

        self.assertDictEqual(expected_manifest, tfe["manifest"])
        self.assertDictEqual(expected_tracks, tfe["tracks"])
        self.assertDictEqual(expected_times, tfe["times"])
        self.assertDictEqual(expected_feature_a, tfe["features"]["feature_a"])
        self.assertDictEqual(expected_feature_b, tfe["features"]["feature_b"])
        self.assertDictEqual(expected_feature_c, tfe["features"]["feature_c"])


if __name__ == "__main__":
    unittest.main()
