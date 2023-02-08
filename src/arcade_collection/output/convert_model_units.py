from typing import Optional, Union

import pandas as pd
from prefect import task


@task
def convert_model_units(
    data: pd.DataFrame, ds: float, dt: float, regions: Optional[Union[list[str], str]] = None
) -> None:
    data["time"] = dt * data["TICK"]
    data["volume"] = ds * ds * ds * data["NUM_VOXELS"]
    data["height"] = ds * (data["MAX_Z"] - data["MIN_Z"] + 1)

    if regions is None:
        return

    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if region == "DEFAULT":
            continue

        data[f"volume.{region}"] = ds * ds * ds * data[f"NUM_VOXELS.{region}"]
        data[f"height.{region}"] = ds * (data[f"MAX_Z.{region}"] - data[f"MIN_Z.{region}"])
