import json
from os import path
import tarfile
from typing import List, Union

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


def convert_state_to_string(state_index: int) -> Union[str, None]:
    """
    Convert the numbers that represent cell state into an annotation.

    Parameters
    ----------
    state_index :
        The index of cell states.

    Returns
    -------
    :
        The cell state annotation.
    """
    if state_index == 0:
        return "NEU"
    elif state_index == 1:
        return "APO"
    elif state_index == 2:
        return "QUI"
    elif state_index == 3:
        return "MIG"
    elif state_index == 4:
        return "PRO"
    elif state_index == 5:
        return "SEN"
    elif state_index == 6:
        return "NEC"

    return None


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
        u = int(location[0])
        v = int(location[1])
        w = int(location[2])
        z = int(location[3])

        for cell in cells:
            population = cell[1]
            state = cell[2]
            position = cell[3]
            volume = cell[4]
            if len(cell[5]) == 0:
                cycle = None
            else:
                cycle = np.mean(cell[5])
            data_list = [
                time,
                seed,
                u,
                v,
                w,
                z,
                position,
                population,
                convert_state_to_string(state),
                volume,
                cycle,
            ]

            parsed_data.append(data_list)

    return parsed_data
