import random
import tarfile
from typing import Optional, Union

import numpy as np
import pandas as pd
from simulariumio import (
    DISPLAY_TYPE,
    AgentData,
    CameraData,
    DimensionData,
    DisplayData,
    MetaData,
    ModelMetaData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def convert_to_simularium(
    series_key: str,
    cells_data_tar: tarfile.TarFile,
    locations_data_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: float,
    dt: float,
    phase_colors: dict[str, str],
    resolution: Optional[int] = None,
) -> str:
    length, width, height = box
    frames = list(np.arange(*frame_spec))

    data = format_tar_data(series_key, cells_data_tar, locations_data_tar, frames, resolution)

    meta_data = get_meta_data(series_key, length, width, height, ds)
    agent_data = get_agent_data(data)
    agent_data.display_data = get_display_data(data, phase_colors)

    for index, (frame, group) in enumerate(data.groupby("frame")):
        n_agents = len(group)
        agent_data.times[index] = float(frame) * dt
        agent_data.n_agents[index] = n_agents
        agent_data.unique_ids[index][:n_agents] = range(0, n_agents)
        agent_data.types[index][:n_agents] = group["name"]
        agent_data.radii[index][:n_agents] = group["radius"]
        agent_data.positions[index][:n_agents, 0] = (group["x"] - length / 2.0) * ds
        agent_data.positions[index][:n_agents, 1] = (width / 2.0 - group["y"]) * ds
        agent_data.positions[index][:n_agents, 2] = (group["z"] - height / 2.0) * ds

    return TrajectoryConverter(
        TrajectoryData(
            meta_data=meta_data,
            agent_data=agent_data,
            time_units=UnitData("hr"),
            spatial_units=UnitData("um"),
        )
    ).to_JSON()


def get_meta_data(series_key: str, length: int, width: int, height: int, ds: float) -> MetaData:
    meta_data = MetaData(
        box_size=np.array([length * ds, width * ds, height * ds]),
        camera_defaults=CameraData(
            position=np.array([10.0, 0.0, 200.0]),
            look_at_position=np.array([10.0, 0.0, 0.0]),
            fov_degrees=60.0,
        ),
        trajectory_title=f"ARCADE - {series_key}",
        model_meta_data=ModelMetaData(
            title="ARCADE",
            version="3.0",
            description=(f"Agent-based modeling framework ARCADE for {series_key}."),
        ),
    )

    return meta_data


def get_agent_data(data: pd.DataFrame) -> AgentData:
    total_frames = len(data["frame"].unique())
    max_agents = data.groupby("frame")["name"].count().max()
    return AgentData.from_dimensions(DimensionData(total_frames, max_agents))


def get_display_data(data: pd.DataFrame, phase_colors: dict[str, str]) -> DisplayData:
    display_data = {}

    for name in data["name"].unique():
        _, cell_id, phase = name.split("#")

        random.seed(cell_id)
        jitter = (random.random() - 0.5) / 2

        display_data[name] = DisplayData(
            name=name,
            display_type=DISPLAY_TYPE.SPHERE,
            color=shade_color(phase_colors[phase], jitter),
        )

    return display_data


def format_tar_data(
    series_key: str,
    cells_tar: tarfile.TarFile,
    locs_tar: tarfile.TarFile,
    frames: list[int],
    resolution: Optional[int],
) -> pd.DataFrame:
    data: list[list[Union[int, str, float]]] = []

    for frame in frames:
        cells = extract_tick_json(cells_tar, series_key, frame, "CELLS")
        locations = extract_tick_json(locs_tar, series_key, frame, "LOCATIONS")

        for cell, location in zip(cells, locations):
            regions = [loc["region"] for loc in location["location"]]

            for region in regions:
                name = f"{region}#{cell['id']}#{cell['phase']}"

                all_voxels = get_location_voxels(location, region if region != "DEFAULT" else None)
                all_voxels = [(x, y, z) for x, y, z in all_voxels]

                if resolution is None:
                    radius = (len(all_voxels) ** (1.0 / 3)) / 1.5
                    center = list(np.array(all_voxels).mean(axis=0))
                    data = data + [[name, int(frame), radius] + center]
                else:
                    radius = resolution / 2
                    center_offset = (resolution - 1) / 2

                    resolution_voxels = get_resolution_voxels(all_voxels, resolution)
                    border_voxels = filter_border_voxels(set(resolution_voxels), resolution)
                    center_voxels = [
                        [x + center_offset, y + center_offset, z + center_offset]
                        for x, y, z in border_voxels
                    ]

                    data = data + [[name, int(frame), radius] + voxel for voxel in center_voxels]

    return pd.DataFrame(data, columns=["name", "frame", "radius", "x", "y", "z"])


def get_resolution_voxels(
    voxels: list[tuple[int, int, int]], resolution: int
) -> list[tuple[int, int, int]]:
    df = pd.DataFrame(voxels, columns=["x", "y", "z"])

    min_x, min_y, min_z = df.min()
    max_x, max_y, max_z = df.max()

    samples = [
        (sx, sy, sz)
        for sx in np.arange(min_x, max_x, resolution)
        for sy in np.arange(min_y, max_y, resolution)
        for sz in np.arange(min_z, max_z, resolution)
    ]

    offsets = [
        (dx, dy, dz)
        for dx in range(resolution)
        for dy in range(resolution)
        for dz in range(resolution)
    ]

    resolution_voxels = []

    for sx, sy, sz in samples:
        sample_voxels = [(sx + dx, sy + dy, sz + dz) for dx, dy, dz in offsets]

        if len(set(sample_voxels) - set(voxels)) < len(offsets) / 2:
            resolution_voxels.append((sx, sy, sz))

    return resolution_voxels


def filter_border_voxels(
    voxels: set[tuple[int, int, int]], resolution: int
) -> list[tuple[int, int, int]]:
    offsets = [
        (resolution, 0, 0),
        (-resolution, 0, 0),
        (0, resolution, 0),
        (0, -resolution, 0),
        (0, 0, resolution),
        (0, 0, -resolution),
    ]
    filtered_voxels = []

    for x, y, z in voxels:
        neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in offsets]
        if len(set(neighbors) - set(voxels)) != 0:
            filtered_voxels.append((x, y, z))

    return filtered_voxels


def shade_color(color: str, alpha: float) -> str:
    old_color = color.replace("#", "")
    old_red, old_green, old_blue = [int(old_color[i : i + 2], 16) for i in (0, 2, 4)]
    layer_color = 0 if alpha < 0 else 255

    new_red = round(old_red + (layer_color - old_red) * abs(alpha))
    new_green = round(old_green + (layer_color - old_green) * abs(alpha))
    new_blue = round(old_blue + (layer_color - old_blue) * abs(alpha))

    return f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
