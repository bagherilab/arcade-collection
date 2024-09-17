import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from arcade_collection.convert.convert_to_meshes import MAX_ARRAY_LEVEL, MeshType, convert_to_meshes

from .utilities import build_tar_instance


def mock_marching_cubes(array, **_):
    array[array != MAX_ARRAY_LEVEL] = 0
    voxels = list(zip(*np.nonzero(array)))
    vertices = np.array(voxels, dtype="float")
    faces = np.reshape(range(len(voxels) * 3), (-1, 3))
    normals = np.array(voxels, dtype="float")
    return vertices, faces, normals, None


class TestConvertToMeshes(unittest.TestCase):
    def setUp(self):
        self.series_key = "SERIES_KEY"
        self.frame_spec = (5, 16, 5)
        self.regions = ["DEFAULT", "REGION"]
        self.box = (6, 6, 3)

        contents = {
            f"{self.series_key}_000005.LOCATIONS.json": [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 1, 1], [1, 2, 1], [2, 2, 1]]},
                        {"region": "REGION", "voxels": [[2, 1, 1]]},
                    ],
                },
                {
                    "id": 2,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 1, 1], [1, 2, 1]]},
                        {"region": "REGION", "voxels": [[2, 1, 1], [2, 2, 1]]},
                    ],
                },
            ],
            f"{self.series_key}_000010.LOCATIONS.json": [
                {
                    "id": 3,
                    "location": [
                        {"region": "DEFAULT", "voxels": []},
                        {
                            "region": "REGION",
                            "voxels": [[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1]],
                        },
                    ],
                },
            ],
            f"{self.series_key}_000015.LOCATIONS.json": [
                {
                    "id": 4,
                    "location": [
                        {"region": "DEFAULT", "voxels": []},
                        {"region": "REGION", "voxels": []},
                    ],
                },
            ],
        }

        self.locations_tar = build_tar_instance(contents)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_meshes"],
        "measure",
        return_value=mock.Mock(),
    )
    def test_convert_to_meshes_default_no_group(self, measure_mock):
        mesh_type = MeshType.DEFAULT
        group_size = None
        categories = None

        measure_mock.marching_cubes.side_effect = mock_marching_cubes

        obj_05_1_default = (
            "v -0.5 -0.5 0.0\n"
            "v -0.5 0.5 0.0\n"
            "v 0.5 -0.5 0.0\n"
            "v 0.5 0.5 0.0\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n"
        )
        obj_05_1_region = "v 0.0 0.0 0.0\nvn 2.0 4.0 1.0\nf 3//3 2//2 1//1\n"
        obj_05_2_default = obj_05_1_default
        obj_05_2_region = (
            "v 0.0 -0.5 0.0\n"
            "v 0.0 0.5 0.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
        )
        obj_10_3_default = obj_05_1_default
        obj_10_3_region = obj_05_1_default

        expected_meshes = [
            (5, 1, "DEFAULT", obj_05_1_default),
            (5, 2, "DEFAULT", obj_05_2_default),
            (5, 1, "REGION", obj_05_1_region),
            (5, 2, "REGION", obj_05_2_region),
            (10, 3, "DEFAULT", obj_10_3_default),
            (10, 3, "REGION", obj_10_3_region),
        ]

        meshes = convert_to_meshes(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            mesh_type,
            group_size,
            categories,
        )

        self.assertCountEqual(expected_meshes, meshes)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_meshes"],
        "measure",
        return_value=mock.Mock(),
    )
    def test_convert_to_meshes_all_mesh_type_no_group(self, measure_mock):
        mesh_type = MeshType.INVERTED
        group_size = None
        categories = None

        measure_mock.marching_cubes.side_effect = mock_marching_cubes

        obj_05_1_default = (
            "v -0.5 -0.5 0.0\n"
            "v -0.5 0.5 0.0\n"
            "v 0.5 -0.5 0.0\n"
            "v 0.5 0.5 0.0\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 1//1 2//2 3//3\n"
            "f 4//4 5//5 6//6\n"
            "f 7//7 8//8 9//9\n"
            "f 10//10 11//11 12//12\n"
        )
        obj_05_1_region = "v 0.0 0.0 0.0\nvn 2.0 4.0 1.0\nf 1//1 2//2 3//3\n"
        obj_05_2_default = obj_05_1_default
        obj_05_2_region = (
            "v 0.0 -0.5 0.0\n"
            "v 0.0 0.5 0.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 1//1 2//2 3//3\n"
            "f 4//4 5//5 6//6\n"
        )
        obj_10_3_default = obj_05_1_default
        obj_10_3_region = obj_05_1_default

        expected_meshes = [
            (5, 1, "DEFAULT", obj_05_1_default),
            (5, 2, "DEFAULT", obj_05_2_default),
            (5, 1, "REGION", obj_05_1_region),
            (5, 2, "REGION", obj_05_2_region),
            (10, 3, "DEFAULT", obj_10_3_default),
            (10, 3, "REGION", obj_10_3_region),
        ]

        meshes = convert_to_meshes(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            mesh_type,
            group_size,
            categories,
        )

        self.assertCountEqual(expected_meshes, meshes)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_meshes"],
        "measure",
        return_value=mock.Mock(),
    )
    def test_convert_to_meshes_region_mesh_type_no_group(self, measure_mock):
        mesh_type = {"DEFAULT": MeshType.DEFAULT, "REGION": MeshType.INVERTED}
        group_size = None
        categories = None

        measure_mock.marching_cubes.side_effect = mock_marching_cubes

        obj_05_1_default = (
            "v -0.5 -0.5 0.0\n"
            "v -0.5 0.5 0.0\n"
            "v 0.5 -0.5 0.0\n"
            "v 0.5 0.5 0.0\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n"
        )
        obj_05_1_region = "v 0.0 0.0 0.0\nvn 2.0 4.0 1.0\nf 1//1 2//2 3//3\n"
        obj_05_2_default = obj_05_1_default
        obj_05_2_region = (
            "v 0.0 -0.5 0.0\n"
            "v 0.0 0.5 0.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 1//1 2//2 3//3\n"
            "f 4//4 5//5 6//6\n"
        )
        obj_10_3_default = obj_05_1_default
        obj_10_3_region = (
            "v -0.5 -0.5 0.0\n"
            "v -0.5 0.5 0.0\n"
            "v 0.5 -0.5 0.0\n"
            "v 0.5 0.5 0.0\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 1//1 2//2 3//3\n"
            "f 4//4 5//5 6//6\n"
            "f 7//7 8//8 9//9\n"
            "f 10//10 11//11 12//12\n"
        )

        expected_meshes = [
            (5, 1, "DEFAULT", obj_05_1_default),
            (5, 2, "DEFAULT", obj_05_2_default),
            (5, 1, "REGION", obj_05_1_region),
            (5, 2, "REGION", obj_05_2_region),
            (10, 3, "DEFAULT", obj_10_3_default),
            (10, 3, "REGION", obj_10_3_region),
        ]

        meshes = convert_to_meshes(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            mesh_type,
            group_size,
            categories,
        )

        self.assertCountEqual(expected_meshes, meshes)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_meshes"],
        "measure",
        return_value=mock.Mock(),
    )
    def test_convert_to_meshes_default_mesh_type_with_group_same_category(self, measure_mock):
        mesh_type = MeshType.DEFAULT
        group_size = 2
        categories = pd.DataFrame(
            {"FRAME": [5, 5, 10, 15], "CATEGORY": ["A", "A", "A", "A"], "ID": [1, 2, 3, 4]}
        )

        measure_mock.marching_cubes.side_effect = mock_marching_cubes

        obj_05_0_default = (
            "v -2.0 0.0 -0.5\n"
            "v -2.0 1.0 -0.5\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n\n"
            "v -2.0 0.0 -0.5\n"
            "v -2.0 1.0 -0.5\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 7//7 6//6 5//5\n"
            "f 10//10 9//9 8//8\n"
            "f 13//13 12//12 11//11\n"
            "f 16//16 15//15 14//14\n"
        )
        obj_05_0_region = (
            "v -1.0 1.0 -0.5\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 4//4 3//3 2//2\n"
            "f 7//7 6//6 5//5\n"
        )
        obj_10_0_default = (
            "v -2.0 0.0 -0.5\n"
            "v -2.0 1.0 -0.5\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n"
        )
        obj_10_0_region = obj_10_0_default

        expected_meshes = [
            (5, 0, "DEFAULT", obj_05_0_default),
            (5, 0, "REGION", obj_05_0_region),
            (10, 0, "DEFAULT", obj_10_0_default),
            (10, 0, "REGION", obj_10_0_region),
        ]

        meshes = convert_to_meshes(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            mesh_type,
            group_size,
            categories,
        )

        self.assertCountEqual(expected_meshes, meshes)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_meshes"],
        "measure",
        return_value=mock.Mock(),
    )
    def test_convert_to_meshes_default_mesh_type_with_group_different_category(self, measure_mock):
        mesh_type = MeshType.DEFAULT
        group_size = 2
        categories = pd.DataFrame(
            {"FRAME": [5, 5, 10, 15], "CATEGORY": ["A", "B", "A", "A"], "ID": [1, 2, 3, 4]}
        )

        measure_mock.marching_cubes.side_effect = mock_marching_cubes

        obj_05_0_default = (
            "v -2.0 0.0 -0.5\n"
            "v -2.0 1.0 -0.5\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n"
        )
        obj_05_0_region = "v -1.0 1.0 -0.5\nvn 2.0 4.0 1.0\nf 3//3 2//2 1//1\n"
        obj_05_1_default = obj_05_0_default
        obj_05_1_region = (
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
        )
        obj_10_0_default = (
            "v -2.0 0.0 -0.5\n"
            "v -2.0 1.0 -0.5\n"
            "v -1.0 0.0 -0.5\n"
            "v -1.0 1.0 -0.5\n"
            "vn 1.0 3.0 1.0\n"
            "vn 1.0 4.0 1.0\n"
            "vn 2.0 3.0 1.0\n"
            "vn 2.0 4.0 1.0\n"
            "f 3//3 2//2 1//1\n"
            "f 6//6 5//5 4//4\n"
            "f 9//9 8//8 7//7\n"
            "f 12//12 11//11 10//10\n"
        )
        obj_10_0_region = obj_10_0_default

        expected_meshes = [
            (5, 0, "DEFAULT", obj_05_0_default),
            (5, 1, "DEFAULT", obj_05_1_default),
            (5, 0, "REGION", obj_05_0_region),
            (5, 1, "REGION", obj_05_1_region),
            (10, 0, "DEFAULT", obj_10_0_default),
            (10, 0, "REGION", obj_10_0_region),
        ]

        meshes = convert_to_meshes(
            self.series_key,
            self.locations_tar,
            self.frame_spec,
            self.regions,
            self.box,
            mesh_type,
            group_size,
            categories,
        )

        self.assertCountEqual(expected_meshes, meshes)


if __name__ == "__main__":
    unittest.main()
