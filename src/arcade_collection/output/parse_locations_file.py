from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tarfile


LOCATIONS_COLUMNS = [
    "ID",
    "TICK",
    "CENTER_X",
    "CENTER_Y",
    "CENTER_Z",
    "MIN_X",
    "MIN_Y",
    "MIN_Z",
    "MAX_X",
    "MAX_Y",
    "MAX_Z",
]
"""Column names for locations data parsed into tidy data format."""


def parse_locations_file(tar: tarfile.TarFile, regions: list[str]) -> pd.DataFrame:
    """
    Parse simulation locations data into tidy data format.

    Parameters
    ----------
    tar
        Tar archive containing locations data.
    regions
        List of regions.

    Returns
    -------
    :
        Parsed locations data.
    """

    all_locations: list[list[str | int]] = []

    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)

        if extracted_member is None:
            continue

        tick = int(member.name.replace(".LOCATIONS.json", "").split("_")[-1])
        locations_json = json.loads(extracted_member.read().decode("utf-8"))

        locations = [parse_location_tick(tick, cell, regions) for cell in locations_json]
        all_locations = all_locations + locations

    columns = LOCATIONS_COLUMNS + [
        f"{column}.{region}" for region in regions for column in LOCATIONS_COLUMNS[2:]
    ]
    return pd.DataFrame(all_locations, columns=columns)


def parse_location_tick(tick: int, location: dict, regions: list[str]) -> list:
    """
    Parse location data for a single simulation tick.

    Original data is formatted as:

    .. code-block:: python

        {
            "id": cell_id,
            "center": [center_x, center_y, center_z],
            "location": [
                {
                    "region": region,
                    "voxels": [
                        [x, y, z],
                        [x, y, z],
                        ...
                    ]
                },
                {
                    "region": region,
                    "voxels": [
                        [x, y, z],
                        [x, y, z],
                        ...
                    ]
                },
                ...
            ]
        }

    Parsed data is formatted as:

    .. code-block:: python

        [ cell_id, tick, center_x, center_y, center_z, min_x, min_y, min_z, max_x, max_y, max_z ]

    When regions are specified, each list also contains centers, minimums, and
    maximums for the corresponding regions.

    Parameters
    ----------
    tick
        Simulation tick.
    location
        Original location data.
    regions
        List of regions.

    Returns
    -------
    :
        Parsed location data.
    """

    if "center" in location:
        voxels = np.array([voxel for region in location["location"] for voxel in region["voxels"]])
        mins = np.min(voxels, axis=0)
        maxs = np.max(voxels, axis=0)
        parsed = [location["id"], tick, *location["center"], *mins, *maxs]
    else:
        parsed = [location["id"], tick, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    for reg in regions:
        region_voxels = np.array(
            [
                voxel
                for region in location["location"]
                for voxel in region["voxels"]
                if region["region"] == reg
            ]
        )

        if len(region_voxels) == 0:
            parsed = [*parsed, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            continue

        center = [int(value + 0.5) for value in region_voxels.mean(axis=0)]
        mins = np.min(region_voxels, axis=0)
        maxs = np.max(region_voxels, axis=0)
        parsed = [*parsed, *center, *mins, *maxs]

    return parsed
