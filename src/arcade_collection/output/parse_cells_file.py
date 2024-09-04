from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import tarfile

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
"""Column names for cells data parsed into tidy data format."""


def parse_cells_file(tar: tarfile.TarFile, regions: list[str]) -> pd.DataFrame:
    """
    Parse simulation cells data into tidy data format.

    Parameters
    ----------
    tar
        Tar archive containing locations data.
    regions
        List of regions.

    Returns
    -------
    :
        Parsed cells data.
    """

    all_cells: list[list[str | int]] = []

    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)

        if extracted_member is None:
            continue

        tick = int(member.name.replace(".CELLS.json", "").split("_")[-1])
        cells_json = json.loads(extracted_member.read().decode("utf-8"))

        cells = [parse_cell_tick(tick, cell, regions) for cell in cells_json]
        all_cells = all_cells + cells

    columns = CELLS_COLUMNS + [f"NUM_VOXELS.{region}" for region in regions]
    return pd.DataFrame(all_cells, columns=columns)


def parse_cell_tick(tick: int, cell: dict, regions: list[str]) -> list:
    """
    Parse cell data for a single simulation tick.

    Original data is formatted as:

    .. code-block:: python

        {
            "id": cell_id,
            "parent": parent_id,
            "pop": population,
            "age": age,
            "divisions": divisions,
            "state": state,
            "phase": phase,
            "voxels": voxels,
            "criticals": [critical_volume, critical_height],
            "regions": [
                {
                    "region": region_name,
                    "voxels": region_voxels,
                    "criticals": [critical_region_volume, critical_region_height]
                },
                ...
            ]
        }

    Parsed data is formatted as:

    .. code-block:: python

        [ cell_id, tick, parent_id, population, age, divisions, state, phase, voxels ]

    When regions are specified, each list also contains the number of voxels for
    the corresponding regions.

    Parameters
    ----------
    tick
        Simulation tick.
    cell
        Original cell data.
    regions
        List of regions.

    Returns
    -------
    :
        Parsed cell data.
    """

    features = ["parent", "pop", "age", "divisions", "state", "phase", "voxels"]
    parsed = [cell["id"], tick] + [cell[feature] for feature in features]

    if regions and "regions" in cell:
        region_voxels = [
            cell_region["voxels"]
            for region in regions
            for cell_region in cell["regions"]
            if cell_region["region"] == region
        ]
        parsed = parsed + region_voxels

    return parsed
