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
    Target column  Source column(s)      Conversion
    =============  ===================  =============================
    ``time``       ``TICK``             ``dt * TICK``
    ``volume``     ``NUM_VOXELS``       ``ds * ds * ds * NUM_VOXELS``
    ``height``     ``MAX_Z`` ``MIN_Z``  ``ds * (MAX_Z - MIN_Z + 1)``
    ``cx``         ``CX``               ``ds * CX``
    ``cy``         ``CY``               ``ds * CY``
    ``cz``         ``CZ``               ``ds * CZ``
    =============  ===================  =============================

    For each region (other than ``DEFAULT``), the following columns are added to the data:

    =================  =================================  ==========================================
    Target column      Source column(s)                   Conversion
    =================  =================================  ==========================================
    ``volume.REGION``  ``NUM_VOXELS.REGION``              ``ds * ds * ds * NUM_VOXELS.REGION``
    ``height.REGION``  ``MAX_Z.REGION`` ``MIN_Z.REGION``  ``ds * (MAX_Z.REGION - MIN_Z.REGION + 1)``
    =================  =================================  ==========================================

    The following property columns are rescaled:

    =====================  =====================  ==========================
    Target column          Source column(s)        Conversion
    =====================  =====================  ==========================
    ``area``               ``area``               ``ds * ds * area``
    ``perimeter``          ``perimeter``          ``ds * perimeter``
    ``axis_major_length``  ``axis_major_length``  ``ds * axis_major_length``
    ``axis_minor_length``  ``axis_minor_length``  ``ds * axis_minor_length``
    =====================  =====================  ==========================

    Parameters
    ----------
    data
        Simulation data.
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

    convert_temporal_units(data, dt)
    convert_spatial_units(data, ds)

    if regions is None:
        return

    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if region == "DEFAULT":
            continue

        convert_spatial_units(data, ds, region)


def convert_temporal_units(data: pd.DataFrame, dt: float) -> None:
    """
    Converts temporal data from simulation units to true units.

    Simulations use temporal unit of ticks. Temporal resolution (hours/tick) is
    used to convert data to true units.

    The following temporal columns are converted:

    =============  ===================  =============================
    Target column  Source column(s)      Conversion
    =============  ===================  =============================
    ``time``       ``TICK``             ``dt * TICK``
    =============  ===================  =============================

    Parameters
    ----------
    data
        Simulation data.
    dt
        Temporal resolution in hours/tick.
    """

    if "TICK" in data.columns:
        data["time"] = round(dt * data["TICK"], 2)


def convert_spatial_units(data: pd.DataFrame, ds: float, region: Optional[str] = None) -> None:
    """
    Converts spatial data from simulation units to true units.

    Simulations use spatial unit of voxels. Spatial resolution (microns/voxel)
    is used to convert data to true units.

    The following spatial columns are converted:

    =====================  =====================  =============================
    Target column          Source column(s)        Conversion
    =====================  =====================  =============================
    ``volume``             ``NUM_VOXELS``         ``ds * ds * ds * NUM_VOXELS``
    ``height``             ``MAX_Z`` ``MIN_Z``    ``ds * (MAX_Z - MIN_Z + 1)``
    ``cx``                 ``CX``                 ``ds * CX``
    ``cy``                 ``CY``                 ``ds * CY``
    ``cz``                 ``CZ``                 ``ds * CZ``
    ``area``               ``area``               ``ds * ds * area``
    ``perimeter``          ``perimeter``          ``ds * perimeter``
    ``axis_major_length``  ``axis_major_length``  ``ds * axis_major_length``
    ``axis_minor_length``  ``axis_minor_length``  ``ds * axis_minor_length``
    =====================  =====================  =============================

    Note that the centroid columns (``cx``, ``cy``, and ``cz``) are only
    converted for the entire cell (``region == None``).

    Parameters
    ----------
    data
        Simulation data.
    ds
        Spatial resolution in microns/voxel.
    region
        Name of region.
    """

    suffix = "" if region is None else f".{region}"

    if f"NUM_VOXELS{suffix}" in data.columns:
        data[f"volume{suffix}"] = ds * ds * ds * data[f"NUM_VOXELS{suffix}"]

    if f"MAX_Z{suffix}" in data.columns and f"MIN_Z{suffix}" in data.columns:
        data[f"height{suffix}"] = ds * (data[f"MAX_Z{suffix}"] - data[f"MIN_Z{suffix}"] + 1)

    if "CX" in data.columns and region is None:
        data["cx"] = ds * data["CX"]

    if "CY" in data.columns and region is None:
        data["cy"] = ds * data["CY"]

    if "CZ" in data.columns and region is None:
        data["cz"] = ds * data["CZ"]

    property_conversions = [
        ("area", ds * ds),
        ("perimeter", ds),
        ("axis_major_length", ds),
        ("axis_minor_length", ds),
    ]

    for name, conversion in property_conversions:
        column = f"{name}{suffix}"

        if column not in data.columns:
            continue

        data[column] = data[column] * conversion


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
