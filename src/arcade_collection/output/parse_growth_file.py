import json
import tarfile

import numpy as np
import pandas as pd

GROWTH_COLUMNS = [
    "TICK",
    "SEED",
    "U",
    "V",
    "W",
    "Z",
    "POSITION",
    "POPULATION",
    "STATE",
    "VOLUME",
    "CYCLE",
]
"""Column names for growth data parsed into tidy data format."""

CELL_STATES = [
    "NEUTRAL",
    "APOPTOTIC",
    "QUIESCENT",
    "MIGRATORY",
    "PROLIFERATIVE",
    "SENESCENT",
    "NECROTIC",
]
"""Cell state names."""


def parse_growth_file(tar: tarfile.TarFile) -> pd.DataFrame:
    """
    Parse simulation growth data into tidy data format.

    Parameters
    ----------
    tar
        Tar archive containing growth data.

    Returns
    -------
    :
        Parsed growth data.
    """

    all_timepoints = []

    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)
        extracted_json = json.loads(extracted_member.read().decode("utf-8"))

        seed = extracted_json["seed"]
        all_timepoints.extend(
            [
                data
                for timepoint in extracted_json["timepoints"]
                for data in parse_growth_timepoint(timepoint, seed)
            ]
        )

    return pd.DataFrame(all_timepoints, columns=GROWTH_COLUMNS)


def parse_growth_timepoint(data: dict, seed: int) -> list:
    """
    Parse growth data for a single simulation timepoint.

    Original data is formatted as:

    .. code-block:: python

        {
            "time": time,
            "cells": [
                [
                    [u, v, w, z],
                    [
                        [
                            type,
                            population,
                            state,
                            position,
                            volume,
                            [cell, cycle, lengths, ...]
                        ],
                        ...
                    ]
                ],
                ...
            ]
        }

    Parsed data is formatted as:

    .. code-block:: python

        [
            [time, seed, u, v, w, z, position, population, state, volume, cell_cycle],
            [time, seed, u, v, w, z, position, population, state, volume, cell_cycle],
            ...
        ]

    Cell cycle length is ``None`` if the cell has not yet divided. Otherwise,
    cell cycle is the average of all cell cycle lengths.

    Parameters
    ----------
    data
        Original simulation data.
    seed
        Random seed.

    Returns
    -------
    :
        Parsed simulation data.
    """

    parsed_data = []
    time = data["time"]

    for location, cells in data["cells"]:
        for cell in cells:
            _, population, state, position, volume, cycles = cell
            cycle = None if len(cycles) == 0 else np.mean(cycles)

            data_list = [
                time,
                seed,
                *location,
                position,
                population,
                CELL_STATES[state],
                volume,
                cycle,
            ]

            parsed_data.append(data_list)

    return parsed_data
