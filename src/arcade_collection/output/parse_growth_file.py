from typing import List, Union
import json
import tarfile

import pandas as pd
from prefect import task


@task
def parse_growth_file(tar: tarfile.TarFile) -> pd.DataFrame:
    all_timepoints = []
    for member in tar.getmembers():
        seed = 0


def parse_growth_timepoint():
    time_index = self.timepoints.index(timepoint)

    parsed_data = []
    sim_timepoint = loaded_simulation["timepoints"][time_index]["cells"]
    param_timepoint = loaded_param_simulation["timepoints"][time_index]["cells"]

    for (location, cells), (_, param_cells) in zip(sim_timepoint, param_timepoint):
        u = int(location[0])
        v = int(location[1])
        w = int(location[2])
        z = int(location[3])
        szudzik_coordinate = self.get_szudzik_pair(u, v)

        for cell, param_cell in zip(cells, param_cells):
            population = cell[1]
            state = cell[2]
            position = cell[3]
            volume = np.round(cell[4])
            cycle = np.round(np.mean(cell[5]))
            max_height = param_cell[4][3]
            meta_pref = param_cell[4][8]
            migra_threshold = param_cell[4][9]

            data_list = [
                self.key,
                self.seed,
                timepoint,
                szudzik_coordinate,
                u,
                v,
                w,
                z,
                position,
                str(population),
                str(state),
                volume,
                cycle,
                max_height,
                meta_pref,
                migra_threshold,
            ]

            parsed_data.append(data_list)

    columns = [feature.name for feature in self.get_feature_list()]
    return pd.DataFrame(parsed_data, columns=columns)
