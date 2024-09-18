from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def convert_to_tfe(
    all_data: pd.DataFrame, features: list[tuple[str, str, str]], frame_spec: tuple[int, int, int]
) -> dict:
    """
    Generate TFE manifest and feature data for simulation.

    Parameters
    ----------
    all_data
        Simulation data containing ID, TICK, and time.
    features
        List of feature keys, names, and data types.
    frame_spec
        Specification for frames.

    Returns
    -------
    :
        TFE manifest and feature data
    """

    frames = list(np.arange(*frame_spec))
    manifest = get_manifest_data(features, frames)

    frame_data = all_data[all_data["TICK"].isin(frames)]

    tracks = get_tracks_from_data(frame_data)
    times = get_times_from_data(frame_data)

    tfe_json = {"manifest": manifest, "tracks": tracks, "times": times, "features": {}}

    for index, (key, _, dtype) in enumerate(features):
        if dtype == "categorical":
            categories = list(all_data[key].unique())
            manifest["features"][index]["categories"] = categories
        else:
            categories = None

        tfe_json["features"][key] = get_feature_from_data(frame_data, key, categories)

    return tfe_json


def get_manifest_data(features: list[tuple[str, str, str]], frames: list[int]) -> dict:
    """
    Build manifest for TFE.

    Parameters
    ----------
    features
        List of feature keys, names, and data types.
    frames
        List of frames.

    Returns
    -------
    :
        Manifest in TFE format.
    """

    return {
        "frames": [f"frames/frame_{i}.png" for i in range(len(frames))],
        "features": [
            {"key": key, "name": name, "data": f"features/{key}.json", "type": dtype}
            for key, name, dtype in features
        ],
        "tracks": "tracks.json",
        "times": "times.json",
    }


def get_tracks_from_data(data: pd.DataFrame) -> dict:
    """
    Extract track ids from data and format for TFE.

    Parameters
    ----------
    data
        Simulation data for selected frames.

    Returns
    -------
    :
        Track data in TFE format.
    """

    return {"data": [0, *list(data["ID"])]}


def get_times_from_data(data: pd.DataFrame) -> dict:
    """
    Extract time points from data and format for TFE.

    Parameters
    ----------
    data
        Simulation data for selected frames.

    Returns
    -------
    :
        Time data in TFE format.
    """

    return {"data": [0, *list(data["time"])]}


def get_feature_from_data(data: pd.DataFrame, feature: str, categories: list | None = None) -> dict:
    """
    Extract specified feature from data and format for TFE.

    Parameters
    ----------
    data
        Simulation data for selected frames.
    feature
        Feature key.
    categories
        List of data categories (if data is categorical).

    Returns
    -------
    :
        Feature data in TFE format.
    """

    if categories is not None:
        feature_values = data[feature].apply(categories.index)
    else:
        feature_values = data[feature]

    feature_min = float(np.nanmin(feature_values))
    feature_max = float(np.nanmax(feature_values))

    return {"data": [0, *list(feature_values)], "min": feature_min, "max": feature_max}
