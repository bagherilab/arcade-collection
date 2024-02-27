import re
from typing import Optional, Union

import pandas as pd


def convert_model_units(
    data: pd.DataFrame,
    ds: Optional[float],
    dt: Optional[float],
    regions: Optional[Union[list[str], str]] = None,
) -> None:
    if dt is None:
        dt = data["KEY"].apply(estimate_temporal_resolution)

    if ds is None:
        ds = data["KEY"].apply(estimate_spatial_resolution)

    data["time"] = round(dt * data["TICK"], 2)

    if "NUM_VOXELS" in data.columns:
        data["volume"] = ds * ds * ds * data["NUM_VOXELS"]

    if "MAX_Z" in data.columns and "MIN_Z" in data.columns:
        data["height"] = ds * (data["MAX_Z"] - data["MIN_Z"] + 1)

    if "CX" in data.columns:
        data["cx"] = ds * data["CX"]

    if "CY" in data.columns:
        data["cy"] = ds * data["CY"]

    if "CZ" in data.columns:
        data["cz"] = ds * data["CZ"]

    if regions is None:
        return

    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if region == "DEFAULT":
            continue

        data[f"volume.{region}"] = ds * ds * ds * data[f"NUM_VOXELS.{region}"]
        data[f"height.{region}"] = ds * (data[f"MAX_Z.{region}"] - data[f"MIN_Z.{region}"])


def estimate_temporal_resolution(key: str) -> float:
    match = re.findall(r"[_]?DT([0-9]+)[_]?", key)

    if len(match) == 1:
        return float(match[0]) / 60

    return 1.0


def estimate_spatial_resolution(key: str) -> float:
    match = re.findall(r"[_]?DS([0-9]+)[_]?", key)

    if len(match) == 1:
        return float(match[0])

    return 1.0
