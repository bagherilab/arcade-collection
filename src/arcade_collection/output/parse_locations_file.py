from typing import List, Union
import json
import tarfile

import numpy as np
import pandas as pd
from prefect import task

CELLS_COLUMNS = [
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


@task
def parse_locations_file(tar: tarfile.TarFile, regions: list[str]) -> pd.DataFrame:
    all_locations: List[List[Union[str, int]]] = []

    for member in tar.getmembers():
        timepoint = int(member.name.replace(".LOCATIONS.json", "").split("_")[-1])

        extracted_member = tar.extractfile(member)
        assert extracted_member is not None
        locations_json = json.loads(extracted_member.read().decode("utf-8"))

        locations = [parse_location_timepoint(timepoint, cell, regions) for cell in locations_json]
        all_locations = all_locations + locations

    columns = CELLS_COLUMNS + [
        f"{column}.{region}" for region in regions for column in CELLS_COLUMNS[2:]
    ]
    locations_df = pd.DataFrame(all_locations, columns=columns)

    return locations_df


def parse_location_timepoint(timepoint: int, loc: dict, regions: list[str]) -> list:
    if "center" in loc:
        voxels = np.array([voxel for region in loc["location"] for voxel in region["voxels"]])
        mins = np.min(voxels, axis=0)
        maxs = np.max(voxels, axis=0)
        parsed = [loc["id"], timepoint, *loc["center"], *mins, *maxs]
    else:
        parsed = [loc["id"], timepoint, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    for reg in regions:
        region_voxels = np.array(
            [
                voxel
                for region in loc["location"]
                for voxel in region["voxels"]
                if region["region"] == reg
            ]
        )

        if len(region_voxels) == 0:
            parsed = parsed + [-1, -1, -1, -1, -1, -1, -1, -1, -1]
            continue

        center = [round(value) for value in region_voxels.mean(axis=0)]
        mins = np.min(region_voxels, axis=0)
        maxs = np.max(region_voxels, axis=0)
        parsed = parsed + [*center, *mins, *maxs]

    return parsed
