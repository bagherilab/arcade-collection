import random
import unittest

import numpy as np
import pandas as pd

from arcade_collection.output.convert_model_units import (
    convert_model_units,
    convert_spatial_units,
    convert_temporal_units,
    estimate_spatial_resolution,
    estimate_temporal_resolution,
)


class TestConvertModelUnits(unittest.TestCase):
    def test_convert_model_units_no_estimate_no_regions(self):
        dt = 1.0 / 3
        ds = 1.0 / 7
        regions = None
        ticks = np.random.randint(100, size=10)
        voxels = np.random.randint(100, size=10)
        center = np.random.randint(100, size=(10, 3))
        z = np.random.randint(100, size=(10, 2))

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "TICK": ticks,
                "NUM_VOXELS": voxels,
                "CENTER_X": center[:, 0],
                "CENTER_Y": center[:, 1],
                "CENTER_Z": center[:, 2],
                "MIN_Z": z[:, 0],
                "MAX_Z": z[:, 1],
            }
        )

        expected = data.copy()
        expected["time"] = np.round(ticks * dt, 2)
        expected["volume"] = ds * ds * ds * voxels
        expected["height"] = ds * (z[:, 1] - z[:, 0] + 1)
        expected["cx"] = ds * center[:, 0]
        expected["cy"] = ds * center[:, 1]
        expected["cz"] = ds * center[:, 2]

        convert_model_units(data, ds, dt, regions)

        self.assertTrue(expected.equals(data))

    def test_convert_model_units_with_estimate_no_regions(self):
        dt = None
        ds = None
        regions = None
        ticks = np.random.randint(100, size=10)
        voxels = np.random.randint(100, size=10)
        center = np.random.randint(100, size=(10, 3))
        z = np.random.randint(100, size=(10, 2))

        all_dt = np.random.randint(1, 10, size=10)
        all_ds = np.random.randint(1, 10, size=10)
        dt = all_dt / 60
        ds = all_ds.astype("float")

        data = pd.DataFrame(
            {
                "KEY": [f"DT{dt:03d}_DS{ds:03d}" for dt, ds in zip(all_dt, all_ds)],
                "PLACEHOLDER": np.random.rand(10),
                "TICK": ticks,
                "NUM_VOXELS": voxels,
                "CENTER_X": center[:, 0],
                "CENTER_Y": center[:, 1],
                "CENTER_Z": center[:, 2],
                "MIN_Z": z[:, 0],
                "MAX_Z": z[:, 1],
            }
        )

        expected = data.copy()
        expected["time"] = np.round(ticks * dt, 2)
        expected["volume"] = ds * ds * ds * voxels
        expected["height"] = ds * (z[:, 1] - z[:, 0] + 1)
        expected["cx"] = ds * center[:, 0]
        expected["cy"] = ds * center[:, 1]
        expected["cz"] = ds * center[:, 2]

        convert_model_units(data, None, None, regions)

        self.assertTrue(expected.equals(data))

    def test_convert_model_units_no_estimate_single_region(self):
        dt = 1.0 / 3
        ds = 1.0 / 7
        regions = "REGION"
        ticks = np.random.randint(100, size=10)
        voxels = np.random.randint(100, size=10)
        region_voxels = np.random.randint(100, size=10)

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "TICK": ticks,
                "NUM_VOXELS": voxels,
                f"NUM_VOXELS.{regions}": region_voxels,
            }
        )

        expected = data.copy()
        expected["time"] = np.round(ticks * dt, 2)
        expected["volume"] = ds * ds * ds * voxels
        expected[f"volume.{regions}"] = ds * ds * ds * region_voxels

        convert_model_units(data, ds, dt, regions)

        self.assertTrue(expected.equals(data))

    def test_convert_model_units_no_estimate_multiple_regions(self):
        dt = 1.0 / 3
        ds = 1.0 / 7
        regions = ["DEFAULT", "REGION_A", "REGION_B"]
        ticks = np.random.randint(100, size=10)
        voxels = np.random.randint(100, size=10)
        region_voxels = np.random.randint(100, size=(10, 2))

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "TICK": ticks,
                "NUM_VOXELS": voxels,
                f"NUM_VOXELS.{regions[1]}": region_voxels[:, 0],
                f"NUM_VOXELS.{regions[2]}": region_voxels[:, 1],
            }
        )

        expected = data.copy()
        expected["time"] = np.round(ticks * dt, 2)
        expected["volume"] = ds * ds * ds * voxels
        expected[f"volume.{regions[1]}"] = ds * ds * ds * region_voxels[:, 0]
        expected[f"volume.{regions[2]}"] = ds * ds * ds * region_voxels[:, 1]

        convert_model_units(data, ds, dt, regions)

        self.assertTrue(expected.equals(data))

    def test_convert_temporal_units_no_columns(self):
        dt = 1.0 / 3

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
            }
        )

        expected = data.copy()

        convert_temporal_units(data, dt)

        self.assertTrue(expected.equals(data))

    def test_convert_temporal_units_with_columns(self):
        dt = 1.0 / 3
        ticks = np.random.randint(100, size=10)

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "TICK": ticks,
            }
        )

        expected = data.copy()
        expected["time"] = np.round(ticks * dt, 2)

        convert_temporal_units(data, dt)

        self.assertTrue(expected.equals(data))

    def test_convert_spatial_units_no_columns(self):
        ds = 1.0 / 3
        region = None

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
            }
        )

        expected = data.copy()

        convert_spatial_units(data, ds, region)

        self.assertTrue(expected.equals(data))

    def test_convert_spatial_units_with_columns_no_region_no_properties(self):
        ds = 1.0 / 3
        region = None
        voxels = np.random.randint(100, size=10)
        center = np.random.randint(100, size=(10, 3))
        z = np.random.randint(100, size=(10, 2))

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "NUM_VOXELS": voxels,
                "CENTER_X": center[:, 0],
                "CENTER_Y": center[:, 1],
                "CENTER_Z": center[:, 2],
                "MIN_Z": z[:, 0],
                "MAX_Z": z[:, 1],
            }
        )

        expected = data.copy()
        expected["volume"] = ds * ds * ds * voxels
        expected["height"] = ds * (z[:, 1] - z[:, 0] + 1)
        expected["cx"] = ds * center[:, 0]
        expected["cy"] = ds * center[:, 1]
        expected["cz"] = ds * center[:, 2]

        convert_spatial_units(data, ds, region)

        self.assertTrue(expected.equals(data))

    def test_convert_spatial_units_with_columns_no_region_with_properties(self):
        ds = 1.0 / 3
        region = None
        area = np.random.rand(10)
        perimeter = np.random.rand(10)
        axis_major_length = np.random.rand(10)
        axis_minor_length = np.random.rand(10)

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "area": area,
                "perimeter": perimeter,
                "axis_major_length": axis_major_length,
                "axis_minor_length": axis_minor_length,
            }
        )

        expected = data.copy()
        expected["area"] = ds * ds * area
        expected["perimeter"] = ds * perimeter
        expected["axis_major_length"] = ds * axis_major_length
        expected["axis_minor_length"] = ds * axis_minor_length

        convert_spatial_units(data, ds, region)

        self.assertTrue(expected.equals(data))

    def test_convert_spatial_units_with_columns_with_region_no_properties(self):
        ds = 1.0 / 3
        region = "REGION"
        voxels = np.random.randint(100, size=10)
        z = np.random.randint(100, size=(10, 2))

        data = pd.DataFrame(
            {
                "PLACEHOLDER": np.random.rand(10),
                "NUM_VOXELS": np.random.randint(100, size=10),
                f"NUM_VOXELS.{region}": voxels,
                "CENTER_X": np.random.randint(100, size=10),
                "CENTER_Y": np.random.randint(100, size=10),
                "CENTER_Z": np.random.randint(100, size=10),
                "MIN_Z": np.random.randint(100, size=10),
                "MAX_Z": np.random.randint(100, size=10),
                f"MIN_Z.{region}": z[:, 0],
                f"MAX_Z.{region}": z[:, 1],
            }
        )

        expected = data.copy()
        expected[f"volume.{region}"] = ds * ds * ds * voxels
        expected[f"height.{region}"] = ds * (z[:, 1] - z[:, 0] + 1)

        convert_spatial_units(data, ds, region)

        self.assertTrue(expected.equals(data))

    def test_convert_spatial_units_with_columns_with_region_with_properties(self):
        ds = 1.0 / 3
        region = "REGION"
        placeholder = np.random.rand(10)
        area = np.random.rand(10)
        perimeter = np.random.rand(10)
        axis_major_length = np.random.rand(10)
        axis_minor_length = np.random.rand(10)

        data = pd.DataFrame(
            {
                "PLACEHOLDER": placeholder,
                "area": area,
                "perimeter": perimeter,
                "axis_major_length": axis_major_length,
                "axis_minor_length": axis_minor_length,
                f"area.{region}": area,
                f"perimeter.{region}": perimeter,
                f"axis_major_length.{region}": axis_major_length,
                f"axis_minor_length.{region}": axis_minor_length,
            }
        )

        expected = pd.DataFrame(
            {
                "PLACEHOLDER": placeholder,
                "area": area,
                "perimeter": perimeter,
                "axis_major_length": axis_major_length,
                "axis_minor_length": axis_minor_length,
                f"area.{region}": ds * ds * area,
                f"perimeter.{region}": ds * perimeter,
                f"axis_major_length.{region}": ds * axis_major_length,
                f"axis_minor_length.{region}": ds * axis_minor_length,
            }
        )

        convert_spatial_units(data, ds, region)

        self.assertTrue(expected.equals(data))

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

    def test_estimate_spatial_resolution_invalid_spatial_key(self):
        ds = int(random.random() * 10)
        ds_key = f"DS{ds:03d}x"

        self.assertEqual(1, estimate_spatial_resolution(f"{ds_key}_B_C"))
        self.assertEqual(1, estimate_spatial_resolution(f"A_{ds_key}_C"))
        self.assertEqual(1, estimate_spatial_resolution(f"A_B_{ds_key}"))


if __name__ == "__main__":
    unittest.main()
