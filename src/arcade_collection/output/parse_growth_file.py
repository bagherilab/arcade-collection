import json
import tarfile

import numpy as np
import pandas as pd
from prefect import task

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


@task
def parse_growth_file(tar: tarfile.TarFile) -> pd.DataFrame:
    """
    Parse the tumor growth tar file.

    Parameters
    ----------
    tar :
        Tar file of simulations.

    Returns
    -------
    :
        Data of all timepoints of all simulations in tar file.
    """
    all_timepoints = []
    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)
        if extracted_member is not None:
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
    Parse one timepoint of the simulation.

    The original data contains data of every timepoint at a seed in a
    dictionary. The current data contains data of one cell per row, with tick,
    seed, coordinates (u, v, w, z), position, population, state, volume, and
    averaged cycle.

    Parameters
    ----------
    timepoint :
        The data of one timepoint.

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
