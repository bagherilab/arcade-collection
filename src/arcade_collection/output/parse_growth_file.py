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

CELL_STATES = [
    "NEUTRAL",
    "APOPTOTIC",
    "QUIESCENT",
    "MIGRATORY",
    "PROLIFERATIVE",
    "SENESCENT",
    "NECROTIC",
]


def parse_growth_file(tar: tarfile.TarFile) -> pd.DataFrame:
    """
    Parses a tumor growth simulation tar file.

    Parameters
    ----------
    tar :
        Tar file of simulations for different seeds.

    Returns
    -------
    :
        Parsed simulation data for all seeds and timepoints.
    """

    all_timepoints = []

    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)
        assert extracted_member is not None
        extracted_json = json.loads(extracted_member.read().decode("utf-8"))

        seed = extracted_json["seed"]
        all_timepoints.extend(
            [
                data
                for timepoint in extracted_json["timepoints"]
                for data in parse_growth_timepoint(timepoint, seed)
            ]
        )

    timepoints_df = pd.DataFrame(all_timepoints, columns=GROWTH_COLUMNS)

    return timepoints_df


def parse_growth_timepoint(timepoint: dict, seed: int) -> list:
    """
    Parses a simulation timepoint into a list of features per cell.

    The original data contains cell features in the form:

    .. code-block:: text

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

    Parsed data is formatted into:

    .. code-block:: text

        [
            [time, seed, u, v, w, z, position, population, state, volume, cycle],
            [time, seed, u, v, w, z, position, population, state, volume, cycle],
            ...
        ]

    Cell cycle length is `None` if the cell has not yet divided. Otherwise, cell
    cycle is the average of all cell cycle lengths.

    Parameters
    ----------
    timepoint :
        Data for a timepoint.

    Returns
    -------
    :
        Parsed data of the timepoint.
    """

    parsed_data = []
    time = timepoint["time"]

    for (location, cells) in timepoint["cells"]:
        u, v, w, z = location

        for cell in cells:
            _, population, state, position, volume, cycles = cell

            if len(cycles) == 0:
                cycle = None
            else:
                cycle = np.mean(cycles)

            data_list = [
                time,
                seed,
                u,
                v,
                w,
                z,
                position,
                population,
                CELL_STATES[state],
                volume,
                cycle,
            ]

            parsed_data.append(data_list)

    return parsed_data
