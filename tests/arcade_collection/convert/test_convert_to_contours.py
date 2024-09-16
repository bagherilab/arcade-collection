import unittest

from arcade_collection.convert.convert_to_contours import convert_to_contours

from .utilities import build_tar_instance


class TestConvertToContours(unittest.TestCase):
    def setUp(self):
        self.series_key = "SERIES_KEY"
        self.frame = 5
        self.file = f"{self.series_key}_{self.frame:06d}.LOCATIONS.json"

    def test_convert_to_contours_no_voxels(self):
        regions = ["DEFAULT"]
        box = (1, 1, 1)
        indices = {"top": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": []},
                    ],
                },
            ]
        }

        data_tar = build_tar_instance(contents)

        expected_contours = {"DEFAULT": {"top": {}}}

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)

    def test_convert_to_contours_no_index(self):
        regions = ["DEFAULT"]
        box = (3, 3, 3)
        indices = {"top": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[0, 0, 2]]},
                    ],
                },
            ]
        }

        data_tar = build_tar_instance(contents)

        expected_contours = {"DEFAULT": {"top": {}}}

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)

    def test_convert_to_contours_different_views(self):
        regions = ["DEFAULT"]
        box = (4, 5, 6)
        indices = {"top": [1], "side1": [1], "side2": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {
                            "region": "DEFAULT",
                            "voxels": [
                                [1, 1, 1],
                                [1, 2, 1],
                                [1, 3, 1],
                                [2, 3, 1],
                                [2, 1, 1],
                                [2, 1, 2],
                                [2, 1, 3],
                                [1, 1, 2],
                            ],
                        },
                    ],
                }
            ]
        }

        expected_contours = {
            "DEFAULT": {
                "top": {
                    1: [
                        [
                            [2.5, 3.0],
                            [2.0, 2.5],
                            [1.5, 2.0],
                            [2.0, 1.5],
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [0.5, 3.0],
                            [1.0, 3.5],
                            [2.0, 3.5],
                            [2.5, 3.0],
                        ]
                    ]
                },
                "side1": {
                    1: [
                        [
                            [2.5, 3.0],
                            [2.5, 2.0],
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [1.0, 2.5],
                            [1.5, 3.0],
                            [2.0, 3.5],
                            [2.5, 3.0],
                        ]
                    ]
                },
                "side2": {
                    1: [
                        [
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [0.5, 3.0],
                            [1.0, 3.5],
                            [1.5, 3.0],
                            [1.5, 2.0],
                            [2.0, 1.5],
                            [2.5, 1.0],
                        ]
                    ]
                },
            }
        }

        data_tar = build_tar_instance(contents)

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)

    def test_convert_to_contours_multiple_regions(self):
        regions = ["DEFAULT", "REGION_A", "REGION_B"]
        box = (4, 5, 6)
        indices = {"top": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 1, 1], [1, 2, 1], [2, 1, 1]]},
                        {"region": "REGION_A", "voxels": [[2, 2, 1]]},
                        {"region": "REGION_B", "voxels": [[2, 3, 1]]},
                    ],
                }
            ]
        }

        expected_contours = {
            "DEFAULT": {
                "top": {
                    1: [
                        [
                            [2.5, 3.0],
                            [2.5, 2.0],
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [1.0, 2.5],
                            [1.5, 3.0],
                            [2.0, 3.5],
                            [2.5, 3.0],
                        ]
                    ]
                }
            },
            "REGION_A": {
                "top": {1: [[[2.5, 2.0], [2.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 2.0]]]}
            },
            "REGION_B": {
                "top": {1: [[[2.5, 3.0], [2.0, 2.5], [1.5, 3.0], [2.0, 3.5], [2.5, 3.0]]]}
            },
        }

        data_tar = build_tar_instance(contents)

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)

    def test_convert_to_contours_disconnected_location(self):
        regions = ["DEFAULT"]
        box = (5, 6, 6)
        indices = {"top": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {
                            "region": "DEFAULT",
                            "voxels": [
                                [1, 1, 1],
                                [1, 2, 1],
                                [2, 1, 1],
                                [3, 3, 1],
                                [3, 4, 1],
                                [2, 4, 1],
                            ],
                        },
                    ],
                },
            ]
        }

        expected_contours = {
            "DEFAULT": {
                "top": {
                    1: [
                        [
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [1.0, 2.5],
                            [1.5, 2.0],
                            [2.0, 1.5],
                            [2.5, 1.0],
                        ],
                        [
                            [3.5, 4.0],
                            [3.5, 3.0],
                            [3.0, 2.5],
                            [2.5, 3.0],
                            [2.0, 3.5],
                            [1.5, 4.0],
                            [2.0, 4.5],
                            [3.0, 4.5],
                            [3.5, 4.0],
                        ],
                    ]
                }
            }
        }

        data_tar = build_tar_instance(contents)

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)

    def test_convert_to_contours_multiple_locations(self):
        regions = ["DEFAULT"]
        box = (4, 5, 6)
        indices = {"top": [1]}
        contents = {
            self.file: [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 1, 1], [1, 2, 1], [2, 1, 1]]},
                    ],
                },
                {
                    "id": 2,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[2, 2, 1], [2, 3, 1], [1, 3, 1]]},
                    ],
                },
            ]
        }

        expected_contours = {
            "DEFAULT": {
                "top": {
                    1: [
                        [
                            [2.5, 1.0],
                            [2.0, 0.5],
                            [1.0, 0.5],
                            [0.5, 1.0],
                            [0.5, 2.0],
                            [1.0, 2.5],
                            [1.5, 2.0],
                            [2.0, 1.5],
                            [2.5, 1.0],
                        ],
                        [
                            [2.5, 3.0],
                            [2.5, 2.0],
                            [2.0, 1.5],
                            [1.5, 2.0],
                            [1.0, 2.5],
                            [0.5, 3.0],
                            [1.0, 3.5],
                            [2.0, 3.5],
                            [2.5, 3.0],
                        ],
                    ]
                }
            }
        }

        data_tar = build_tar_instance(contents)

        contours = convert_to_contours(self.series_key, data_tar, self.frame, regions, box, indices)

        self.assertDictEqual(expected_contours, contours)


if __name__ == "__main__":
    unittest.main()
