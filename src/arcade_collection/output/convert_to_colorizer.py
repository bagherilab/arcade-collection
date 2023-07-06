import numpy as np
import pandas as pd

from arcade_collection.output.convert_model_units import convert_model_units


def convert_to_colorizer(
    all_data: pd.DataFrame,
    features: list[str],
    frame_spec: tuple[int, int, int],
    ds: float,
    dt: float,
    regions: list[str],
) -> dict:
    frames = list(np.arange(*frame_spec))
    manifest = get_manifest_data(features, frames)

    convert_model_units(all_data, ds, dt, regions)
    frame_data = all_data[all_data["TICK"].isin(frames)]

    outliers = get_outliers_from_data(frame_data)
    tracks = get_tracks_from_data(frame_data)
    times = get_times_from_data(frame_data)

    colorizer_json = {
        "manifest": manifest,
        "outliers": outliers,
        "tracks": tracks,
        "times": times,
    }

    for feature in features:
        colorizer_json[feature] = get_feature_from_data(frame_data, feature)

    return colorizer_json


def get_manifest_data(features: list[str], frames: list[int]) -> dict:
    manifest = {
        "frames": [f"frame_{i}.png" for i in range(len(frames))],
        "features": {
            feature: f"feature_{feature_index}.json"
            for feature_index, feature in enumerate(features)
        },
        "outliers": "outliers.json",
        "tracks": "tracks.json",
        "times": "times.json",
    }

    return manifest


def get_outliers_from_data(data: pd.DataFrame) -> dict:
    outliers = [False] * len(data)
    outliers_json = {"data": outliers, "min": False, "max": True}
    return outliers_json


def get_tracks_from_data(data: pd.DataFrame) -> dict:
    tracks = data["ID"]
    tracks_json = {"data": list(tracks)}
    return tracks_json


def get_times_from_data(data: pd.DataFrame) -> dict:
    times = data["time"]
    times_json = {"data": list(times)}
    return times_json


def get_feature_from_data(data: pd.DataFrame, feature: str) -> dict:
    feature_values = data[feature]
    feature_min = float(np.nanmin(feature_values))
    feature_max = float(np.nanmax(feature_values))

    feature_json = {"data": list(feature_values), "min": feature_min, "max": feature_max}
    return feature_json
