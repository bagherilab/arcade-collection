import random
import tarfile
from math import cos, isnan, pi, sin, sqrt
from typing import Optional, Union

import numpy as np
import pandas as pd

from arcade_collection.convert.convert_to_simularium import convert_to_simularium
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


def convert_to_simularium_shapes(
    series_key: str,
    simulation_type: str,
    data_tars: dict[str, tarfile.TarFile],
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: float,
    dz: float,
    dt: float,
    colors: dict[str, str],
    resolution: int = 0,
) -> str:

    if simulation_type == "patch":
        frames = list(map(float, np.arange(*frame_spec)))
        radius, margin, height = box
        bounds = radius + margin
        length = (2 / sqrt(3)) * (3 * (radius + margin) - 1)
        width = 4 * (radius + margin) - 2
        data = format_patch_for_shapes(
            series_key, data_tars["cells"], data_tars["graph"], frames, bounds
        )
    elif simulation_type == "potts":
        frames = list(map(int, np.arange(*frame_spec)))
        length, width, height = box
        data = format_potts_for_shapes(
            series_key, data_tars["cells"], data_tars["locations"], frames, resolution
        )
    else:
        raise ValueError(f"invalid simulation type {simulation_type}")

    return convert_to_simularium(
        series_key, simulation_type, data, length, width, height, ds, dz, dt, colors
    )


def format_patch_for_shapes(
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

                name = f"POPULATION{population}#{cell_id}#{CELL_STATES[state]}"
                radius = (volume ** (1.0 / 3)) / 1.5

                x = (3 * (u + bounds) - 1) / sqrt(3)
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

            for from_node, to_node, edge in graph_timepoint:
                edge_type, radius, _, _, _, _, flow = edge

                name = f"VASCULATURE##{'UNDEFINED' if isnan(flow) else EDGE_TYPES[edge_type + 2]}"

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


def format_potts_for_shapes(
    series_key: str,
    cells_tar: tarfile.TarFile,
    locations_tar: tarfile.TarFile,
    frames: list[Union[int, float]],
    resolution: int,
) -> pd.DataFrame:
    data: list[list[object]] = []

    for frame in frames:
        cells = extract_tick_json(cells_tar, series_key, frame, "CELLS")
        locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

        for cell, location in zip(cells, locations):
            regions = [loc["region"] for loc in location["location"]]

            for region in regions:
                name = f"{region}#{cell['id']}#{cell['phase']}"

                all_voxels = get_location_voxels(location, region if region != "DEFAULT" else None)

                if resolution == 0:
                    radius = (len(all_voxels) ** (1.0 / 3)) / 1.5
                    center = list(np.array(all_voxels).mean(axis=0))
                    data = data + [[name, int(frame), radius] + center + [[]]]
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
