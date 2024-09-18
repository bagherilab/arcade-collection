import numpy as np
import pandas as pd

from arcade_collection.convert.convert_to_simularium import convert_to_simularium


def convert_to_simularium_objects(
    series_key: str,
    simulation_type: str,
    categories: pd.DataFrame,
    regions: list[str],
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: tuple[float, float, float],
    dt: float,
    colors: dict[str, str],
    group_size: int,
    url: str,
    jitter: float = 1.0,
) -> str:
    """
    Convert data to Simularium trajectory using mesh objects.

    Parameters
    ----------
    series_key
        Simulation series key.
    simulation_type : {'potts'}
        Simulation type.
    categories
        Simulation data containing ID, FRAME, and CATEGORY.
    regions
        List of regions.
    frame_spec
        Specification for simulation ticks.
    box
        Size of bounding box.
    ds
        Spatial scaling in um/voxel.
    dt
        Temporal scaling in hours/tick.
    colors
        Map of category to colors.
    group_size
        Number of objects in each mesh group.
    url
        URL for mesh object files.
    jitter
        Relative jitter applied to colors (set to 0 for exact colors).

    Returns
    -------
    :
        Simularium trajectory.
    """

    if simulation_type == "potts":
        frames = list(map(int, np.arange(*frame_spec)))
        length, width, height = box
        data = format_potts_for_objects(
            categories, frames, group_size, regions, length, width, height
        )
    else:
        message = f"invalid simulation type {simulation_type}"
        raise ValueError(message)

    return convert_to_simularium(
        series_key, simulation_type, data, length, width, height, ds, dt, colors, url, jitter
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
    """
    Format ``potts`` simulation data for object-based Simularium trajectory.

    Parameters
    ----------
    categories
        Simulation data containing ID, FRAME, and CATEGORY.
    frames
        List of frames.
    group_size
        Number of objects in each mesh group.
    regions
        List of regions.
    length
        Length of bounding box.
    width
        Width of bounding box.
    height
        Height of bounding box.

    Returns
    -------
    :
        Data formatted for trajectory.
    """

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
                    data = [*data, [name, int(frame), 1, *center, [], "OBJ"]]

            index_offset = index_offset + len(group_ids)

    return pd.DataFrame(
        data, columns=["name", "frame", "radius", "x", "y", "z", "points", "display"]
    )
