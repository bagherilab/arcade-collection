from __future__ import annotations

import random
from math import cos, isnan, pi, sin, sqrt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from arcade_collection.convert.convert_to_simularium import convert_to_simularium
from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels

if TYPE_CHECKING:
    import tarfile

CELL_STATES: list[str] = [
    "UNDEFINED",
    "APOPTOTIC",
    "QUIESCENT",
    "MIGRATORY",
    "PROLIFERATIVE",
    "SENESCENT",
    "NECROTIC",
]
"""Indexed cell states."""

EDGE_TYPES: list[str] = [
    "ARTERIOLE",
    "ARTERY",
    "CAPILLARY",
    "VEIN",
    "VENULE",
    "UNDEFINED",
]
"""Indexed graph edge types."""


def convert_to_simularium_shapes(
    series_key: str,
    simulation_type: str,
    data_tars: dict[str, tarfile.TarFile],
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: tuple[float, float, float],
    dt: float,
    colors: dict[str, str],
    resolution: int = 0,
    jitter: float = 1.0,
) -> str:
    """
    Convert data to Simularium trajectory using shapes.

    Parameters
    ----------
    series_key
        Simulation series key.
    simulation_type : {'patch', 'potts'}
        Simulation type.
    data_tars
        Map of simulation data archives.
    frame_spec
        Specification for simulation ticks.
    box
        Size of bounding box.
    ds
        Spatial scaling in units/um.
    dt
        Temporal scaling in hours/tick.
    colors
        Map of category to colors.
    resolution
        Number of voxels represented by a sphere (0 for single sphere per cell).
    jitter
        Relative jitter applied to colors (set to 0 for exact colors).

    Returns
    -------
    :
        Simularium trajectory.
    """

    # Throw exception if invalid simulation type.
    if simulation_type not in ("patch", "potts"):
        message = f"invalid simulation type {simulation_type}"
        raise ValueError(message)

    if simulation_type == "patch":
        # Simulation type must have either or both "cells" and "graph" data
        if not ("cells" in data_tars or "graph" in data_tars):
            return ""

        frames = list(map(float, np.arange(*frame_spec)))
        radius, margin, height = box
        bounds, length, width = calculate_patch_size(radius, margin)
        data = format_patch_for_shapes(
            series_key, data_tars.get("cells"), data_tars.get("graph"), frames, bounds
        )
    elif simulation_type == "potts":
        # Simulation type must have both "cells" and "locations" data
        if not ("cells" in data_tars and "locations" in data_tars):
            return ""

        frames = list(map(int, np.arange(*frame_spec)))
        length, width, height = box
        data = format_potts_for_shapes(
            series_key, data_tars["cells"], data_tars["locations"], frames, resolution
        )

    return convert_to_simularium(
        series_key, simulation_type, data, length, width, height, ds, dt, colors, jitter=jitter
    )


def format_patch_for_shapes(
    series_key: str,
    cells_tar: tarfile.TarFile | None,
    graph_tar: tarfile.TarFile | None,
    frames: list[float],
    bounds: int,
) -> pd.DataFrame:
    """
    Format ``patch`` simulation data for shape-based Simularium trajectory.

    Parameters
    ----------
    series_key
        Simulation series key.
    cells_tar
        Archive of cell agent data.
    graph_tar
        Archive of vascular graph data.
    frames
        List of frames.
    bounds
        Simulation bounds size (radius + margin).

    Returns
    -------
    :
        Data formatted for trajectory.
    """

    data: list[list[int | str | float | list]] = []

    for frame in frames:
        if cells_tar is not None:
            cell_timepoint = extract_tick_json(cells_tar, series_key, frame, field="cells")

            for location, cells in cell_timepoint:
                u, v, w, z = location
                rotation = random.randint(0, 5)  # noqa: S311

                for cell in cells:
                    _, population, state, position, volume, _ = cell
                    cell_id = f"{u}{v}{w}{z}{position}"

                    name = f"POPULATION{population}#{CELL_STATES[state]}#{cell_id}"
                    radius = float("%.2g" % ((volume ** (1.0 / 3)) / 1.5))  # round to 2 sig figs

                    offset = (position + rotation) % 6
                    x, y = convert_hexagonal_to_rectangular_coordinates((u, v, w), bounds, offset)
                    center = [x, y, z]

                    data = [*data, [name, frame, radius, *center, [], "SPHERE"]]

        if graph_tar is not None:
            graph_timepoint = extract_tick_json(
                graph_tar, series_key, frame, "GRAPH", field="graph"
            )

            for from_node, to_node, edge in graph_timepoint:
                edge_type, radius, _, _, _, _, flow = edge

                name = f"VASCULATURE#{'UNDEFINED' if isnan(flow) else EDGE_TYPES[edge_type + 2]}"

                subpoints = [
                    from_node[0] / sqrt(3),
                    from_node[1],
                    from_node[2],
                    to_node[0] / sqrt(3),
                    to_node[1],
                    to_node[2],
                ]

                data = [*data, [name, frame, radius, 0, 0, 0, subpoints, "FIBER"]]

    return pd.DataFrame(
        data, columns=["name", "frame", "radius", "x", "y", "z", "points", "display"]
    )


def convert_hexagonal_to_rectangular_coordinates(
    uvw: tuple[int, int, int], bounds: int, offset: int
) -> tuple[float, float]:
    """
    Convert hexagonal (u, v, w) coordinates to rectangular (x, y) coordinates.

    Conversion is based on the bounds of the simulation,

    Parameters
    ----------
    uvw
        Hexagonal (u, v, w) coordinates.
    bounds
        Simulation bounds size (radius + margin).
    offset
        Index of hexagonal offset.

    Returns
    -------
    :
        Rectangular (x, y) coordinates.
    """

    u, v, w = uvw
    theta = [pi * (60 * i) / 180.0 for i in range(6)]
    dx = [cos(t) / sqrt(3) for t in theta]
    dy = [sin(t) / sqrt(3) for t in theta]

    x = (3 * (u + bounds) - 1) / sqrt(3)
    y = (v - w) + 2 * bounds - 1

    return x + dx[offset], y + dy[offset]


def calculate_patch_size(radius: int, margin: int) -> tuple[int, float, float]:
    """
    Calculate hexagonal patch simulation sizes.

    Parameters
    ----------
    radius
        Number of hexagonal patches from the center patch.
    margin
        Number of hexagonal patches in the margin.

    Returns
    -------
    :
        Bounds, length, and width of the simulation bounding box.
    """

    bounds = radius + margin
    length = (2 / sqrt(3)) * (3 * bounds - 1)
    width = 4 * bounds - 2

    return bounds, length, width


def format_potts_for_shapes(
    series_key: str,
    cells_tar: tarfile.TarFile,
    locations_tar: tarfile.TarFile,
    frames: list[float],
    resolution: int,
) -> pd.DataFrame:
    """
    Format `potts` simulation data for shape-based Simularium trajectory.

    The resolution parameter can be used to tune how many spheres are used to
    represent each cell. Resolution = 0 displays each cell as a single sphere
    centered on the average voxel position. Resolution = 1 displays each
    individual voxel of each cell as a single sphere.

    Resolution = N will aggregate voxels by dividing the voxels into NxNxN
    cubes, and replacing cubes with at least 50% of those voxels occupied with a
    single sphere centered at the center of the cube.

    For resolution > 0, interior voxels (fully surrounded voxels) are not
    removed.

    Parameters
    ----------
    series_key
        Simulation series key.
    cells_tar
        Archive of cell data.
    locations_tar
        Archive of location data.
    frames
        List of frames.
    resolution
        Number of voxels represented by a sphere (0 for single sphere per cell).

    Returns
    -------
    :
        Data formatted for trajectory.
    """

    data: list[list[object]] = []

    for frame in frames:
        cells = extract_tick_json(cells_tar, series_key, frame, "CELLS")
        locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

        for cell, location in zip(cells, locations):
            regions = [loc["region"] for loc in location["location"]]

            for region in regions:
                name = f"{region}#{cell['phase']}#{cell['id']}"
                all_voxels = get_location_voxels(location, region if region != "DEFAULT" else None)

                if resolution == 0:
                    radius = approximate_radius_from_voxels(len(all_voxels))
                    center = list(np.array(all_voxels).mean(axis=0))
                    data = [*data, [name, int(frame), radius, *center, [], "SPHERE"]]
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
                        [name, int(frame), radius, *voxel, [], "SPHERE"] for voxel in center_voxels
                    ]

    return pd.DataFrame(
        data, columns=["name", "frame", "radius", "x", "y", "z", "points", "display"]
    )


def approximate_radius_from_voxels(voxels: int) -> float:
    """
    Approximate display sphere radius from number of voxels.

    Parameters
    ----------
    voxels
        Number of voxels.

    Returns
    -------
    :
        Approximate radius.
    """

    return (voxels ** (1.0 / 3)) / 1.5


def get_resolution_voxels(
    voxels: list[tuple[int, int, int]], resolution: int
) -> list[tuple[int, int, int]]:
    """
    Get voxels at specified resolution.

    Parameters
    ----------
    voxels
        List of voxels.
    resolution
        Resolution of voxels.

    Returns
    -------
    :
        List of voxels at specified resolution.
    """

    voxel_df = pd.DataFrame(voxels, columns=["x", "y", "z"])

    min_x, min_y, min_z = voxel_df.min()
    max_x, max_y, max_z = voxel_df.max()

    samples = [
        (sx, sy, sz)
        for sx in np.arange(min_x, max_x + 1, resolution)
        for sy in np.arange(min_y, max_y + 1, resolution)
        for sz in np.arange(min_z, max_z + 1, resolution)
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
    """
    Filter voxels to only include the border voxels.

    Parameters
    ----------
    voxels
        List of voxels.
    resolution
        Resolution of voxels.

    Returns
    -------
    :
        List of filtered voxels.
    """

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

    return sorted(filtered_voxels)
