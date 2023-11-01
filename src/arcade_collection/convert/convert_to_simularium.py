import itertools
import random
import tarfile
from math import cos, isnan, pi, sin, sqrt
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

CELL_STATES: list[str] = [
    "UNDEFINED",
    "APOPTOTIC",
    "QUIESCENT",
    "MIGRATORY",
    "PROLIFERATIVE",
    "SENESCENT",
    "NECROTIC",
]

EDGE_TYPES: list[str] = [
    "ARTERIOLE",
    "ARTERY",
    "CAPILLARY",
    "VEIN",
    "VENULE",
    "UNDEFINED",
]


CAMERA_POSITIONS: dict[str, tuple[float, float, float]] = {
    "patch": (0.0, -0.5, 900),
    "potts": (10.0, 0.0, 200.0),
}

CAMERA_LOOK_AT: dict[str, tuple[float, float, float]] = {
    "patch": (0.0, -0.2, 0.0),
    "potts": (10.0, 0.0, 0.0),
}


def convert_to_simularium(
    series_key: str,
    simulation_type: str,
    data_tars: dict[str, tarfile.TarFile],
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: float,
    dz: float,
    dt: float,
    colors: dict[str, str],
    resolution: Optional[int] = None,
    url: Optional[str] = None,
) -> str:

    if simulation_type == "patch":
        frames = list(map(float, np.arange(*frame_spec)))
        radius, margin, height = box
        bounds = radius + margin
        length = (2 / sqrt(3)) * (3 * (radius + margin) - 1)
        width = 4 * (radius + margin) - 2
        data = format_patch_tar_data(
            series_key, data_tars["cells"], data_tars["graph"], frames, bounds
        )
    elif simulation_type == "potts":
        frames = list(map(int, np.arange(*frame_spec)))
        length, width, height = box
        data = format_potts_tar_data(
            series_key, data_tars["cells"], data_tars["locations"], frames, resolution
        )
    else:
        raise ValueError(f"invalid simulation type {simulation_type}")

    meta_data = get_meta_data(series_key, simulation_type, length, width, height, ds, dz)
    agent_data = get_agent_data(data)
    agent_data.display_data = get_display_data(series_key, data, colors, url)

    for index, (frame, group) in enumerate(data.groupby("frame")):
        n_agents = len(group)
        agent_data.times[index] = float(frame) * dt
        agent_data.n_agents[index] = n_agents
        agent_data.unique_ids[index][:n_agents] = range(0, n_agents)
        agent_data.types[index][:n_agents] = group["name"]
        agent_data.radii[index][:n_agents] = group["radius"]
        agent_data.positions[index][:n_agents] = group[["x", "y", "z"]]
        agent_data.n_subpoints[index][:n_agents] = group["points"].map(lambda points: len(points))
        agent_data.viz_types[index][:n_agents] = group["points"].map(
            lambda points: 1001 if points else 1000
        )
        agent_data.subpoints[index][:n_agents] = np.array(
            list(itertools.zip_longest(*group["points"], fillvalue=0))
        ).T

    agent_data.positions[:, :, 0] = (agent_data.positions[:, :, 0] - length / 2.0) * ds
    agent_data.positions[:, :, 1] = (width / 2.0 - agent_data.positions[:, :, 1]) * ds
    agent_data.positions[:, :, 2] = (agent_data.positions[:, :, 2] - height / 2.0) * dz

    agent_data.subpoints[:, :, 0::3] = (agent_data.subpoints[:, :, 0::3]) * ds
    agent_data.subpoints[:, :, 1::3] = (-agent_data.subpoints[:, :, 1::3]) * ds
    agent_data.subpoints[:, :, 2::3] = (agent_data.subpoints[:, :, 2::3]) * dz

    return TrajectoryConverter(
        TrajectoryData(
            meta_data=meta_data,
            agent_data=agent_data,
            time_units=UnitData("hr"),
            spatial_units=UnitData("um"),
        )
    ).to_JSON()


def get_meta_data(
    series_key: str,
    simulation_type: str,
    length: Union[int, float],
    width: Union[int, float],
    height: Union[int, float],
    ds: float,
    dz: float,
) -> MetaData:
    meta_data = MetaData(
        box_size=np.array([length * ds, width * ds, height * dz]),
        camera_defaults=CameraData(
            position=np.array(CAMERA_POSITIONS[simulation_type]),
            look_at_position=np.array(CAMERA_LOOK_AT[simulation_type]),
            fov_degrees=60.0,
        ),
        trajectory_title=f"ARCADE - {series_key}",
        model_meta_data=ModelMetaData(
            title="ARCADE",
            version=simulation_type,
            description=(f"Agent-based modeling framework ARCADE for {series_key}."),
        ),
    )

    return meta_data


def get_agent_data(data: pd.DataFrame) -> AgentData:
    total_frames = len(data["frame"].unique())
    max_agents = data.groupby("frame")["name"].count().max()
    max_subpoints = data["points"].map(lambda points: len(points)).max()
    return AgentData.from_dimensions(DimensionData(total_frames, max_agents, max_subpoints))


def get_display_data(
    series_key: str, data: pd.DataFrame, colors: dict[str, str], url: Optional[str] = None
) -> DisplayData:
    display_data = {}

    for name in data["name"].unique():
        group, cell_id, color_key, frame = name.split("#")

        random.seed(cell_id)
        jitter = (random.random() - 0.5) / 2

        if url is not None:
            display_data[name] = DisplayData(
                name=cell_id,
                display_type=DISPLAY_TYPE.OBJ,
                url=f"{url}/{series_key}_{int(frame):06d}_{int(cell_id):06d}_{group}.MESH.obj",
                color=shade_color(colors[color_key], jitter),
            )
        elif cell_id is None:
            display_data[name] = DisplayData(
                name=group,
                display_type=DISPLAY_TYPE.FIBER,
                color=colors[color_key],
            )
        else:
            display_data[name] = DisplayData(
                name=cell_id,
                display_type=DISPLAY_TYPE.SPHERE,
                color=shade_color(colors[color_key], jitter),
            )

    return display_data


def format_patch_tar_data(
    series_key: str,
    cells_tar: tarfile.TarFile,
    graph_tar: Optional[tarfile.TarFile],
    frames: list[Union[int, float]],
    bounds: int,
) -> pd.DataFrame:
    data: list[list[Union[int, str, float]]] = []

    theta = [pi * (60 * i) / 180.0 for i in range(6)]
    dx = [cos(t) / sqrt(3) for t in theta]
    dy = [sin(t) / sqrt(3) for t in theta]

    for frame in frames:
        cell_timepoint = extract_tick_json(cells_tar, series_key, frame, field="cells")

        for location, cells in cell_timepoint:
            u, v, w, z = location
            rotation = random.randint(0, 5)

            for cell in cells:
                _, population, state, position, volume, _ = cell
                cell_id = f"{u}{v}{w}{z}{position}"

                name = f"POPULATION{population}#{cell_id}#{CELL_STATES[state]}#"
                radius = (volume ** (1.0 / 3)) / 1.5

                x = (u + bounds - 1) * sqrt(3) + 1
                y = (v - w) + 2 * bounds - 1

                center = [
                    (x + dx[(position + rotation) % 6]),
                    (y + dy[(position + rotation) % 6]),
                    z,
                ]

                data = data + [[name, frame, radius] + center + [[]]]

        if graph_tar is not None:
            graph_timepoint = extract_tick_json(
                graph_tar, series_key, frame, "GRAPH", field="graph"
            )

            for (from_node, to_node, edge) in graph_timepoint:
                edge_type, radius, _, _, _, _, flow = edge

                name = f"VASCULATURE##{'UNDEFINED' if isnan(flow) else EDGE_TYPES[edge_type + 2]}#"

                subpoints = [
                    from_node[0] / sqrt(3),
                    from_node[1],
                    from_node[2],
                    to_node[0] / sqrt(3),
                    to_node[1],
                    to_node[2],
                ]

                data = data + [[name, frame, radius] + [0, 0, 0] + [subpoints]]

    return pd.DataFrame(data, columns=["name", "frame", "radius", "x", "y", "z", "points"])


def format_potts_tar_data(
    series_key: str,
    cells_tar: tarfile.TarFile,
    locations_tar: tarfile.TarFile,
    frames: list[Union[int, float]],
    resolution: Optional[int],
) -> pd.DataFrame:
    data: list[list[Union[int, str, float]]] = []

    for frame in frames:
        cells = extract_tick_json(cells_tar, series_key, frame, "CELLS")
        locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

        for cell, location in zip(cells, locations):
            regions = [loc["region"] for loc in location["location"]]

            for region in regions:
                name = f"{region}#{cell['id']}#{cell['phase']}#"

                all_voxels = get_location_voxels(location, region if region != "DEFAULT" else None)

                if resolution is None:
                    radius = (len(all_voxels) ** (1.0 / 3)) / 1.5
                    center = list(np.array(all_voxels).mean(axis=0))
                    data = data + [[name, int(frame), radius] + center + [[]]]
                elif resolution == 0:
                    radius = 1
                    center = list(np.array(all_voxels).mean(axis=0))
                    data = data + [[f"{name}{frame}", int(frame), radius] + center + [[]]]
                else:
                    radius = resolution / 2
                    center_offset = (resolution - 1) / 2

                    resolution_voxels = get_resolution_voxels(all_voxels, resolution)
                    border_voxels = filter_border_voxels(set(resolution_voxels), resolution)
                    center_voxels = [
                        [x + center_offset, y + center_offset, z + center_offset]
                        for x, y, z in border_voxels
                    ]

                    data = data + [
                        [name, int(frame), radius] + voxel + [[]] for voxel in center_voxels
                    ]

    return pd.DataFrame(data, columns=["name", "frame", "radius", "x", "y", "z", "points"])


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
