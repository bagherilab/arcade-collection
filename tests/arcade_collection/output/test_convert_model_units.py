import random
import unittest

from arcade_collection.output.convert_model_units import (
    estimate_spatial_resolution,
    estimate_temporal_resolution,
)


class TestExtractVoxelContours(unittest.TestCase):
    def test_estimate_temporal_resolution_missing_temporal_key(self):
        self.assertEqual(1, estimate_temporal_resolution(""))
        self.assertEqual(1, estimate_temporal_resolution("A"))
        self.assertEqual(1, estimate_temporal_resolution("A_B"))
        self.assertEqual(1, estimate_temporal_resolution("A_B_C"))

    def test_estimate_temporal_resolution_valid_temporal_key(self):
        dt = int(random.random() * 10)
        dt_key = f"DT{dt:03d}"

        self.assertEqual(dt / 60, estimate_temporal_resolution(f"{dt_key}_B_C"))
        self.assertEqual(dt / 60, estimate_temporal_resolution(f"A_{dt_key}_C"))
        self.assertEqual(dt / 60, estimate_temporal_resolution(f"A_B_{dt_key}"))

    def test_estimate_temporal_resolution_invalid_temporal_key(self):
        dt = int(random.random() * 10)
        dt_key = f"DT{dt:03d}x"

        self.assertEqual(1, estimate_temporal_resolution(f"{dt_key}_B_C"))
        self.assertEqual(1, estimate_temporal_resolution(f"A_{dt_key}_C"))
        self.assertEqual(1, estimate_temporal_resolution(f"A_B_{dt_key}"))

    def test_estimate_spatial_resolution_missing_spatial_key(self):
        self.assertEqual(1, estimate_spatial_resolution(""))
        self.assertEqual(1, estimate_spatial_resolution("A"))
        self.assertEqual(1, estimate_spatial_resolution("A_B"))
        self.assertEqual(1, estimate_spatial_resolution("A_B_C"))

    def test_estimate_spatial_resolution_valid_spatial_key(self):
        ds = int(random.random() * 10)
        ds_key = f"DS{ds:03d}"

        self.assertEqual(ds, estimate_spatial_resolution(f"{ds_key}_B_C"))
        self.assertEqual(ds, estimate_spatial_resolution(f"A_{ds_key}_C"))
        self.assertEqual(ds, estimate_spatial_resolution(f"A_B_{ds_key}"))

    def test_estimate_spatial_resolution_invalid_spatiall_key(self):
        ds = int(random.random() * 10)
        ds_key = f"DS{ds:03d}x"

        self.assertEqual(1, estimate_spatial_resolution(f"{ds_key}_B_C"))
        self.assertEqual(1, estimate_spatial_resolution(f"A_{ds_key}_C"))
        self.assertEqual(1, estimate_spatial_resolution(f"A_B_{ds_key}"))


if __name__ == "__main__":
    unittest.main()
