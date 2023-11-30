import numpy as np
import pandas as pd

from arcade_collection.convert.convert_to_simularium import convert_to_simularium


def convert_to_simularium_objects(
    series_key: str,
    simulation_type: str,
    categories: pd.DataFrame,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    ds: float,
    dz: float,
    dt: float,
    colors: dict[str, str],
    group_size: int,
    url: str,
) -> str:
    if simulation_type == "potts":
        frames = list(map(int, np.arange(*frame_spec)))
        length, width, height = box
        data = format_potts_for_objects(
            categories, frames, group_size, regions, length, width, height
        )
    else:
        raise ValueError(f"invalid simulation type {simulation_type}")

    return convert_to_simularium(
        series_key, simulation_type, data, length, width, height, ds, dz, dt, colors, url
    )


def format_potts_for_objects(
    categories: pd.DataFrame,
    frames: list[int],
    group_size: int,
    regions: list[str],
    length: int,
    width: int,
    height: int,
) -> pd.DataFrame:
    data: list[list[object]] = []
    center = [length / 2, width / 2, height / 2]

    for frame in frames:
        frame_categories = categories[categories["FRAME"] == frame]
        index_offset = 0

        for category, category_group in frame_categories.groupby("CATEGORY"):
            ids = list(category_group["ID"].values)
            group_ids = [ids[i : i + group_size] for i in range(0, len(ids), group_size)]

            for i, _ in enumerate(group_ids):
                index = i + index_offset

                for region in regions:
                    name = f"{region}#{category}#{index}#{frame}"
                    data = data + [[name, int(frame), 1] + center + [[]]]

            index_offset = index_offset + len(group_ids)

    return pd.DataFrame(data, columns=["name", "frame", "radius", "x", "y", "z", "points"])
