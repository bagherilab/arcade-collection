import unittest

import numpy as np

from arcade_collection.output.get_location_voxels import get_location_voxels


class TestGetLocationVoxels(unittest.TestCase):
    def test_get_location_voxels_no_region_single_region(self):
        voxels1 = np.random.randint(100, size=(100, 3)).tolist()

        location = {"location": [{"region": "UNDEFINED", "voxels": voxels1}]}

        expected_voxels = [tuple(voxel) for voxel in voxels1]

        voxels = get_location_voxels(location)

        self.assertCountEqual(expected_voxels, voxels)

    def test_get_location_voxels_no_region_multiple_regions(self):
        voxels1 = np.random.randint(100, size=(100, 3)).tolist()
        voxels2 = np.random.randint(100, size=(100, 3)).tolist()

        location = {
            "location": [
                {"region": "REGION1", "voxels": voxels1},
                {"region": "REGION2", "voxels": voxels2},
            ]
        }

        expected_voxels = [tuple(voxel) for voxel in voxels1 + voxels2]

        voxels = get_location_voxels(location)

        self.assertCountEqual(expected_voxels, voxels)

    def test_get_location_voxels_with_region(self):
        voxels1 = np.random.randint(100, size=(100, 3)).tolist()
        voxels2 = np.random.randint(100, size=(100, 3)).tolist()

        location = {
            "location": [
                {"region": "REGION1", "voxels": voxels1},
                {"region": "REGION2", "voxels": voxels2},
            ]
        }

        expected_voxels = [tuple(voxel) for voxel in voxels2]

        voxels = get_location_voxels(location, region="REGION2")

        self.assertCountEqual(expected_voxels, voxels)


if __name__ == "__main__":
    unittest.main()
