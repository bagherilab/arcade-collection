import tarfile
from math import sqrt

import numpy as np
import pandas as pd

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def extract_feature_bins(
    data_tars: dict[str, tarfile.TarFile],
    frame: int,
    scale: float,
) -> pd.DataFrame:
    feature_data: dict[tuple[int, int], list[int]] = {}

    for series_key, data_tar in data_tars.items():
        locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

        for location in locations:
            voxels = get_location_voxels(location)
            array = np.array(voxels)

            volume = len(array)
            height = np.ptp(array[:, 2]) + 1
            positions = np.array(list({(x, y) for x, y in array[:, :2].tolist()}))

            for x, y in positions:
                key = (x, y)

                if key not in feature_data:
                    feature_data[key] = [0, 0, 0]

                feature_data[key][0] += 1
                feature_data[key][1] += volume
                feature_data[key][2] += height

    coordinates = list(feature_data.keys())
    values = list(feature_data.values())

    feature_bins = bin_to_hex(coordinates, values, scale)
    feature_bins_dataframe = pd.DataFrame(
        [[x, y] + list(np.array(v).mean(axis=0)) for (x, y), v in feature_bins.items()],
        columns=["x", "y", "count", "volume", "height"],
    )

    return feature_bins_dataframe


def bin_to_hex(
    coordinates: list[tuple[int, int]], values: list[list[int]], scale: float = 1.0
) -> dict[tuple[float, float], list[list[int]]]:
    bins: dict[tuple[float, float], list[list[int]]] = {}

    for (x, y), v in zip(coordinates, values):
        sx = x / scale
        sy = y / scale

        cx1 = scale * round(sx / sqrt(3)) * sqrt(3)
        cy1 = scale * round(sy)
        dist1 = sqrt((x - cx1) ** 2 + (y - cy1) ** 2)

        cx2 = scale * (round(sx / sqrt(3) - 0.4999) + 0.5) * sqrt(3)
        cy2 = scale * (round(sy - 0.49999) + 0.5)
        dist2 = sqrt((x - cx2) ** 2 + (y - cy2) ** 2)

        if dist1 < dist2:
            center = (cx1, cy1)
        else:
            center = (cx2, cy2)

        if center not in bins:
            bins[center] = []

        bins[center].append(v)

    return bins
