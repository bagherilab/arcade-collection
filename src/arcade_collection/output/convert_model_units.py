import re
from typing import Optional, Union

import pandas as pd


def convert_model_units(
    data: pd.DataFrame,
    ds: Optional[float],
    dt: Optional[float],
    regions: Optional[Union[list[str], str]] = None,
) -> None:
    """
    Converts data from simulation units to true units.

    Simulations use spatial unit of voxels and temporal unit of ticks. Spatial
    resolution (microns/voxel) and temporal resolution (hours/tick) are used to
    convert data to true units. If spatial or temporal resolution is not given,
    they will be estimated from the ``KEY`` column of the data.

    The following columns are added to the data:

    =============  ===================  =============================
    Target column  Source column(s)      Calculation
    =============  ===================  =============================
    ``time``       ``TICK``             ``dt * TICK``
    ``volume``     ``NUM_VOXELS``       ``ds * ds * ds * NUM_VOXELS``
    ``height``     ``MAX_Z`` ``MIN_Z``  ``ds * (MAX_Z - MIN_Z + 1)``
    ``cx``         ``CX``               ``ds * CX``
    ``cy``         ``CY``               ``ds * CY``
    ``cz``         ``CZ``               ``ds * CZ``
    =============  ===================  =============================

    For each region (other than ``DEFAULT``), volume and height are calculated:

    =================  =================================  ==========================================
    Target column      Source column(s)                   Calculation
    =================  =================================  ==========================================
    ``volume.REGION``  ``NUM_VOXELS.REGION``              ``ds * ds * ds * NUM_VOXELS.REGION``
    ``height.REGION``  ``MAX_Z.REGION`` ``MIN_Z.REGION``  ``ds * (MAX_Z.REGION - MIN_Z.REGION + 1)``
    =================  =================================  ==========================================

    Parameters
    ----------
    data
        Parsed simulation data.
    ds
        Spatial resolution in microns/voxel, use None to estimate from keys.
    dt
        Temporal resolution in hours/tick, use None to estimate from keys.
    regions
        List of regions.
    """

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
    """
    Estimates temporal resolution based on condition key.

    If the key contains ``DT##``, where ``##`` denotes the temporal resolution
    in minutes/tick, temporal resolution is estimated from ``##``. Otherwise,
    the default temporal resolution is 1 hours/tick.

    Parameters
    ----------
    key
        Condition key.

    Returns
    -------
    :
        Temporal resolution (hours/tick).
    """

    matches = [re.fullmatch(r"DT([0-9]+)", k) for k in key.split("_")]
    return next((float(match.group(1)) / 60 for match in matches if match is not None), 1.0)


def estimate_spatial_resolution(key: str) -> float:
    """
    Estimates spatial resolution based on condition key.

    If the key contains ``DS##``, where ``##`` denotes the spatial resolution
    in micron/voxel, spatial resolution is estimated from ``##``. Otherwise,
    the default spatial resolution is 1 micron/voxel.

    Parameters
    ----------
    key
        Condition key.

    Returns
    -------
    :
        Spatial resolution (micron/voxel).
    """

    matches = [re.fullmatch(r"DS([0-9]+)", k) for k in key.split("_")]
    return next((float(match.group(1)) for match in matches if match is not None), 1.0)
