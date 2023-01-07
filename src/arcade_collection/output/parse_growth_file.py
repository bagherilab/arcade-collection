from typing import List, Union
import json
import tarfile
from prefect import task

import numpy as np
import pandas as pd
import ntpath
from os import path


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
    all_timepoints = []
    for member in tar.getmembers():
        extracted_member = tar.extractfile(member)
        if extracted_member is not None:
            base = ntpath.basename(member.name)
            base_file = path.splitext(base)
            file_name = base_file[0]
            extension = base_file[1]
            if file_name[0] != "." and extension == ".json":
                extracted_json = json.loads(extracted_member.read().decode("utf-8"))
                seed = extracted_json["seed"]

                for timepoint in extracted_json["timepoints"]:
                    one_timepoint = parse_growth_timepoint(timepoint, seed)
                    for data in one_timepoint:
                        all_timepoints.append(data)

    timepoints_df = pd.DataFrame(all_timepoints, columns=GROWTH_COLUMNS)
    return timepoints_df


def convert_state_to_string(state_index: int) -> str:
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


def parse_growth_timepoint(timepoint: dict, seed: int) -> List[list]:
    parsed_data = []

    for (location, cells) in timepoint["cells"]:
        u = int(location[0])
        v = int(location[1])
        w = int(location[2])
        z = int(location[3])

        for cell in cells:
            population = cell[1]
            state = cell[2]
            position = cell[3]
            volume = np.round(cell[4])
            if len(cell[5]) == 0:
                cycle = -1
            else:
                cycle = np.round(np.mean(cell[5]))
            time = timepoint["time"]
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
