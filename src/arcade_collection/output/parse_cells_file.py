from typing import List, Union
import json
import tarfile

import pandas as pd
from prefect import task

CELLS_COLUMNS = [
    "ID",
    "TICK",
    "PARENT",
    "POPULATION",
    "AGE",
    "DIVISIONS",
    "STATE",
    "PHASE",
    "NUM_VOXELS",
]


@task
def parse_cells_file(tar: tarfile.TarFile, regions: list[str]) -> pd.DataFrame:
    all_cells: List[List[Union[str, int]]] = []

    for member in tar.getmembers():
        timepoint = int(member.name.replace(".CELLS.json", "").split("_")[-1])

        extracted_member = tar.extractfile(member)
        assert extracted_member is not None
        cells_json = json.loads(extracted_member.read().decode("utf-8"))

        cells = [parse_cell_timepoint(timepoint, cell, regions) for cell in cells_json]
        all_cells = all_cells + cells

    columns = CELLS_COLUMNS + [f"NUM_VOXELS.{region}" for region in regions]
    cells_df = pd.DataFrame(all_cells, columns=columns)

    return cells_df


def parse_cell_timepoint(timepoint: int, cell: dict, regions: list[str]) -> list:
    features = ["parent", "pop", "age", "divisions", "state", "phase", "voxels"]
    parsed = [cell["id"], timepoint] + [cell[feature] for feature in features]

    if regions and "regions" in cell:
        region_voxels = [
            cell_region["voxels"]
            for region in regions
            for cell_region in cell["regions"]
            if cell_region["region"] == region
        ]
        parsed = parsed + region_voxels

    return parsed
