import unittest

import numpy as np

from arcade_collection.convert.convert_to_images import ImageType, convert_to_images

from .utilities import build_tar_instance


class TestConvertToImages(unittest.TestCase):
    def setUp(self):
        self.series_key = "SERIES_KEY"
        self.frame_spec = (5, 16, 5)
        self.regions = ["DEFAULT", "REGION"]
        self.box = (8, 8, 3)

        contents = {
            f"{self.series_key}_000005.LOCATIONS.json": [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 1, 1], [2, 1, 1], [2, 2, 1]]},
                        {"region": "REGION", "voxels": [[1, 2, 1]]},
                    ],
                },
                {
                    "id": 2,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[5, 3, 1], [5, 4, 1]]},
                        {"region": "REGION", "voxels": [[6, 3, 1], [6, 4, 1]]},
                    ],
                },
            ],
            f"{self.series_key}_000010.LOCATIONS.json": [
                {
                    "id": 3,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[3, 1, 1]]},
                        {"region": "REGION", "voxels": [[3, 2, 1], [4, 1, 1], [4, 2, 1]]},
                    ],
                },
                {
                    "id": 4,
                    "location": [
                        {
                            "region": "DEFAULT",
                            "voxels": [[1, 3, 1], [2, 3, 1], [1, 4, 1], [2, 4, 1]],
                        },
                        {"region": "REGION", "voxels": []},
                    ],
                },
                {
                    "id": 5,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[3, 5, 1], [4, 5, 1]]},
                        {"region": "REGION", "voxels": [[3, 6, 1], [4, 6, 1]]},
                    ],
                },
            ],
            f"{self.series_key}_000015.LOCATIONS.json": [
                {
                    "id": 6,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[2, 4, 1], [3, 3, 1], [3, 4, 1]]},
                        {"region": "REGION", "voxels": [[2, 3, 1], [4, 3, 1], [4, 4, 1]]},
                    ],
                },
            ],
        }

        self.locations_tar = build_tar_instance(contents)

    def test_convert_to_images_full_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FULL

        chunk_00 = np.array(
            [
                [
                    [
                        [
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [0, 0, 3, 3, 0, 0],
                            [0, 0, 3, 3, 0, 0],
                            [4, 4, 0, 0, 0, 0],
                            [4, 4, 0, 0, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 3, 0, 0],
                            [0, 0, 3, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 6, 6, 6, 0, 0],
                            [0, 6, 6, 6, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 6, 0, 6, 0, 0],
                            [0, 0, 0, 6, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ],
            ]
        )

        expected_images = [(0, 0, chunk_00, None)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertIsNone(chunk[3])

    def test_convert_to_images_full_by_frame_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FULL_BY_FRAME

        chunk_00_05 = np.array(
            [
                [
                    [
                        [
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 2, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ]
            ]
        )
        chunk_00_10 = np.array(
            [
                [
                    [
                        [
                            [0, 0, 3, 3, 0, 0],
                            [0, 0, 3, 3, 0, 0],
                            [4, 4, 0, 0, 0, 0],
                            [4, 4, 0, 0, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 3, 0, 0],
                            [0, 0, 3, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 5, 5, 0, 0],
                        ]
                    ],
                ]
            ]
        )
        chunk_00_15 = np.array(
            [
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 6, 6, 6, 0, 0],
                            [0, 6, 6, 6, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 6, 0, 6, 0, 0],
                            [0, 0, 0, 6, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ]
            ]
        )

        expected_images = [(0, 0, chunk_00_05, 5), (0, 0, chunk_00_10, 10), (0, 0, chunk_00_15, 15)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_flat_rgba_by_frame_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FLAT_RGBA_BY_FRAME

        chunk_00_05 = np.zeros((6, 6, 4), dtype=np.uint8)
        chunk_00_10 = np.zeros((6, 6, 4), dtype=np.uint8)
        chunk_00_15 = np.zeros((6, 6, 4), dtype=np.uint8)

        chunk_00_05[:, :, 3] = 255
        chunk_00_10[:, :, 3] = 255
        chunk_00_15[:, :, 3] = 255

        chunk_00_05[:, :, 0] = [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        chunk_00_10[:, :, 0] = [
            [0, 0, 3, 3, 0, 0],
            [0, 0, 3, 3, 0, 0],
            [4, 4, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 0],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 5, 5, 0, 0],
        ]
        chunk_00_15[:, :, 0] = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0, 0],
            [0, 6, 6, 6, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        expected_images = [(0, 0, chunk_00_05, 5), (0, 0, chunk_00_10, 10), (0, 0, chunk_00_15, 15)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_full_binary_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FULL_BINARY

        chunk_00 = np.array(
            [
                [
                    [
                        [
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                        ]
                    ],
                ],
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ],
            ]
        )

        expected_images = [(0, 0, chunk_00, None)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertIsNone(chunk[3])

    def test_convert_to_images_full_binary_by_frame_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FULL_BINARY_BY_FRAME

        chunk_00_05 = np.array(
            [
                [
                    [
                        [
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ]
            ]
        )
        chunk_00_10 = np.array(
            [
                [
                    [
                        [
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0],
                        ]
                    ],
                ]
            ]
        )
        chunk_00_15 = np.array(
            [
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    ],
                ]
            ]
        )

        expected_images = [(0, 0, chunk_00_05, 5), (0, 0, chunk_00_10, 10), (0, 0, chunk_00_15, 15)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_flat_binary_by_frame_without_chunks(self):
        chunk_size = 6
        image_type = ImageType.FLAT_BINARY_BY_FRAME

        chunk_00_05 = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        chunk_00_10 = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        )
        chunk_00_15 = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        expected_images = [(0, 0, chunk_00_05, 5), (0, 0, chunk_00_10, 10), (0, 0, chunk_00_15, 15)]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_full_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FULL

        chunk_00 = np.array(
            [
                [[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 3], [0, 0, 3], [4, 4, 0]]], [[[0, 0, 0], [0, 0, 3], [0, 0, 0]]]],
                [[[[0, 0, 0], [0, 0, 0], [0, 6, 6]]], [[[0, 0, 0], [0, 0, 0], [0, 6, 0]]]],
            ]
        )
        chunk_10 = np.array(
            [
                [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
                [[[[4, 4, 0], [0, 0, 5], [0, 0, 5]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 5]]]],
                [[[[0, 6, 6], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
            ]
        )
        chunk_01 = np.array(
            [
                [[[[0, 0, 0], [0, 0, 0], [0, 2, 2]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 2]]]],
                [[[[3, 0, 0], [3, 0, 0], [0, 0, 0]]], [[[3, 0, 0], [3, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 0], [0, 0, 0], [6, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [6, 0, 0]]]],
            ]
        )
        chunk_11 = np.array(
            [
                [[[[0, 2, 2], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 2], [0, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 0], [5, 0, 0], [5, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [5, 0, 0]]]],
                [[[[6, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[6, 0, 0], [0, 0, 0], [0, 0, 0]]]],
            ]
        )

        expected_images = [
            (0, 0, chunk_00, None),
            (0, 1, chunk_01, None),
            (1, 0, chunk_10, None),
            (1, 1, chunk_11, None),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertIsNone(chunk[3])

    def test_convert_to_images_full_by_frame_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FULL_BY_FRAME

        chunk_00_05 = np.array(
            [[[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]]]
        )
        chunk_00_10 = np.array(
            [[[[[0, 0, 3], [0, 0, 3], [4, 4, 0]]], [[[0, 0, 0], [0, 0, 3], [0, 0, 0]]]]]
        )
        chunk_00_15 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [0, 6, 6]]], [[[0, 0, 0], [0, 0, 0], [0, 6, 0]]]]]
        )
        chunk_10_10 = np.array(
            [[[[[4, 4, 0], [0, 0, 5], [0, 0, 5]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 5]]]]]
        )
        chunk_10_15 = np.array(
            [[[[[0, 6, 6], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]]
        )
        chunk_01_05 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [0, 2, 2]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 2]]]]]
        )
        chunk_01_10 = np.array(
            [[[[[3, 0, 0], [3, 0, 0], [0, 0, 0]]], [[[3, 0, 0], [3, 0, 0], [0, 0, 0]]]]]
        )
        chunk_01_15 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [6, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [6, 0, 0]]]]]
        )
        chunk_11_05 = np.array(
            [[[[[0, 2, 2], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 2], [0, 0, 0], [0, 0, 0]]]]]
        )
        chunk_11_10 = np.array(
            [[[[[0, 0, 0], [5, 0, 0], [5, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [5, 0, 0]]]]]
        )
        chunk_11_15 = np.array(
            [[[[[6, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[6, 0, 0], [0, 0, 0], [0, 0, 0]]]]]
        )

        expected_images = [
            (0, 0, chunk_00_05, 5),
            (0, 1, chunk_01_05, 5),
            (1, 1, chunk_11_05, 5),
            (0, 0, chunk_00_10, 10),
            (0, 1, chunk_01_10, 10),
            (1, 0, chunk_10_10, 10),
            (1, 1, chunk_11_10, 10),
            (0, 0, chunk_00_15, 15),
            (0, 1, chunk_01_15, 15),
            (1, 0, chunk_10_15, 15),
            (1, 1, chunk_11_15, 15),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_flat_rgba_by_frame_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FLAT_RGBA_BY_FRAME

        chunk_00_05 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_00_10 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_00_15 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_01_05 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_01_10 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_01_15 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_10_10 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_10_15 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_11_05 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_11_10 = np.zeros((3, 3, 4), dtype=np.uint8)
        chunk_11_15 = np.zeros((3, 3, 4), dtype=np.uint8)

        chunk_00_05[:, :, 3] = 255
        chunk_00_10[:, :, 3] = 255
        chunk_00_15[:, :, 3] = 255
        chunk_01_05[:, :, 3] = 255
        chunk_01_10[:, :, 3] = 255
        chunk_01_15[:, :, 3] = 255
        chunk_10_10[:, :, 3] = 255
        chunk_10_15[:, :, 3] = 255
        chunk_11_05[:, :, 3] = 255
        chunk_11_10[:, :, 3] = 255
        chunk_11_15[:, :, 3] = 255

        chunk_00_05[:, :, 0] = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        chunk_00_10[:, :, 0] = [[0, 0, 3], [0, 0, 3], [4, 4, 0]]
        chunk_00_15[:, :, 0] = [[0, 0, 0], [0, 0, 0], [0, 6, 6]]
        chunk_10_10[:, :, 0] = [[4, 4, 0], [0, 0, 5], [0, 0, 5]]
        chunk_10_15[:, :, 0] = [[0, 6, 6], [0, 0, 0], [0, 0, 0]]
        chunk_01_05[:, :, 0] = [[0, 0, 0], [0, 0, 0], [0, 2, 2]]
        chunk_01_10[:, :, 0] = [[3, 0, 0], [3, 0, 0], [0, 0, 0]]
        chunk_01_15[:, :, 0] = [[0, 0, 0], [0, 0, 0], [6, 0, 0]]
        chunk_11_05[:, :, 0] = [[0, 2, 2], [0, 0, 0], [0, 0, 0]]
        chunk_11_10[:, :, 0] = [[0, 0, 0], [5, 0, 0], [5, 0, 0]]
        chunk_11_15[:, :, 0] = [[6, 0, 0], [0, 0, 0], [0, 0, 0]]

        expected_images = [
            (0, 0, chunk_00_05, 5),
            (0, 1, chunk_01_05, 5),
            (1, 1, chunk_11_05, 5),
            (0, 0, chunk_00_10, 10),
            (0, 1, chunk_01_10, 10),
            (1, 0, chunk_10_10, 10),
            (1, 1, chunk_11_10, 10),
            (0, 0, chunk_00_15, 15),
            (0, 1, chunk_01_15, 15),
            (1, 0, chunk_10_15, 15),
            (1, 1, chunk_11_15, 15),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_full_binary_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FULL_BINARY

        chunk_00 = np.array(
            [
                [[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 1], [0, 0, 1], [1, 1, 0]]], [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]]],
                [[[[0, 0, 0], [0, 0, 0], [0, 1, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]]],
            ]
        )
        chunk_10 = np.array(
            [
                [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
                [[[[1, 1, 0], [0, 0, 1], [0, 0, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]],
                [[[[0, 1, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
            ]
        )
        chunk_01 = np.array(
            [
                [[[[0, 0, 0], [0, 0, 0], [0, 1, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]],
                [[[[1, 0, 0], [1, 0, 0], [0, 0, 0]]], [[[1, 0, 0], [1, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 0], [0, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]],
            ]
        )
        chunk_11 = np.array(
            [
                [[[[0, 1, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]]],
                [[[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]],
                [[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]],
            ]
        )

        expected_images = [
            (0, 0, chunk_00, None),
            (0, 1, chunk_01, None),
            (1, 0, chunk_10, None),
            (1, 1, chunk_11, None),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertIsNone(chunk[3])

    def test_convert_to_images_full_binary_by_frame_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FULL_BINARY_BY_FRAME

        chunk_00_05 = np.array(
            [[[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]]]
        )
        chunk_00_10 = np.array(
            [[[[[0, 0, 1], [0, 0, 1], [1, 1, 0]]], [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]]]]
        )
        chunk_00_15 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [0, 1, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]]]]
        )
        chunk_10_10 = np.array(
            [[[[[1, 1, 0], [0, 0, 1], [0, 0, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]]
        )
        chunk_10_15 = np.array(
            [[[[[0, 1, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]]
        )
        chunk_01_05 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [0, 1, 1]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]]
        )
        chunk_01_10 = np.array(
            [[[[[1, 0, 0], [1, 0, 0], [0, 0, 0]]], [[[1, 0, 0], [1, 0, 0], [0, 0, 0]]]]]
        )
        chunk_01_15 = np.array(
            [[[[[0, 0, 0], [0, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]]
        )
        chunk_11_05 = np.array(
            [[[[[0, 1, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]]]]
        )
        chunk_11_10 = np.array(
            [[[[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]]
        )
        chunk_11_15 = np.array(
            [[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]]
        )

        expected_images = [
            (0, 0, chunk_00_05, 5),
            (0, 1, chunk_01_05, 5),
            (1, 1, chunk_11_05, 5),
            (0, 0, chunk_00_10, 10),
            (0, 1, chunk_01_10, 10),
            (1, 0, chunk_10_10, 10),
            (1, 1, chunk_11_10, 10),
            (0, 0, chunk_00_15, 15),
            (0, 1, chunk_01_15, 15),
            (1, 0, chunk_10_15, 15),
            (1, 1, chunk_11_15, 15),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])

    def test_convert_to_images_flat_binary_by_frame_with_chunks(self):
        chunk_size = 3
        image_type = ImageType.FLAT_BINARY_BY_FRAME

        chunk_00_05 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        chunk_00_10 = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
        chunk_00_15 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        chunk_10_10 = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]])
        chunk_10_15 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        chunk_01_05 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        chunk_01_10 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        chunk_01_15 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        chunk_11_05 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        chunk_11_10 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
        chunk_11_15 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        expected_images = [
            (0, 0, chunk_00_05, 5),
            (0, 1, chunk_01_05, 5),
            (1, 1, chunk_11_05, 5),
            (0, 0, chunk_00_10, 10),
            (0, 1, chunk_01_10, 10),
            (1, 0, chunk_10_10, 10),
            (1, 1, chunk_11_10, 10),
            (0, 0, chunk_00_15, 15),
            (0, 1, chunk_01_15, 15),
            (1, 0, chunk_10_15, 15),
            (1, 1, chunk_11_15, 15),
        ]

        images = convert_to_images(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            chunk_size,
            image_type,
        )

        for expected_chunk, chunk in zip(expected_images, images):
            self.assertEqual(expected_chunk[0], chunk[0])
            self.assertEqual(expected_chunk[1], chunk[1])
            self.assertTrue(np.array_equal(expected_chunk[2], chunk[2]))
            self.assertEqual(expected_chunk[3], chunk[3])


if __name__ == "__main__":
    unittest.main()
