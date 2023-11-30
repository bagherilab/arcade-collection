import tarfile
from typing import Optional, Union

import numpy as np
import pandas as pd
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def convert_to_meshes(
    series_key: str,
    data_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    invert: Union[bool, dict[str, bool]] = False,
    group_size: Optional[int] = None,
    categories: Optional[pd.DataFrame] = None,
) -> list[tuple[int, int, str, str]]:

    frames = list(np.arange(*frame_spec))
    meshes = []

    length, width, height = box

    if group_size is not None:
        groups = make_mesh_groups(categories, frames, group_size)
    else:
        groups = None

    for frame in frames:
        locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

        for region in regions:
            region_invert = invert[region] if isinstance(invert, dict) else invert

            if groups is None:
                for location in locations:
                    location_id = location["id"]
                    mesh = make_individual_mesh(
                        location, length, width, height, region, region_invert
                    )

                    if mesh is None:
                        continue

                    meshes.append((frame, location_id, region, mesh))
            else:
                for index, group in groups[frame].items():
                    group_locations = [
                        location for location in locations if location["id"] in group
                    ]
                    mesh = make_combined_mesh(
                        group_locations, length, width, height, region, region_invert
                    )

                    if mesh is None:
                        continue

                    meshes.append((frame, index, region, mesh))

    return meshes


def make_mesh_groups(
    categories: pd.DataFrame, frames: list[int], group_size: int
) -> dict[int, dict[int, list[int]]]:
    groups: dict[int, dict[int, list[int]]] = {}

    for frame in frames:
        groups[frame] = {}
        frame_categories = categories[categories["FRAME"] == frame]
        index_offset = 0

        for _, category_group in frame_categories.groupby("CATEGORY"):
            ids = list(category_group["ID"].values)
            group_ids = [ids[i : i + group_size] for i in range(0, len(ids), group_size)]
            groups[frame].update({i + index_offset: group for i, group in enumerate(group_ids)})

            index_offset = index_offset + len(group_ids)

    return groups


def make_individual_mesh(
    location: dict, length: int, width: int, height: int, region: str, invert: bool
) -> Optional[str]:
    voxels = [
        (x, width - y - 1, z)
        for x, y, z in get_location_voxels(location, region if region != "DEFAULT" else None)
    ]

    if len(voxels) == 0:
        return None

    center = list(np.array(voxels).mean(axis=0))
    array = make_mesh_array(voxels, length, width, height)
    verts, faces, normals = make_mesh_geometry(array, center)
    mesh = make_mesh_file(verts, faces, normals, invert)

    return mesh


def make_combined_mesh(
    locations: list[dict], length: int, width: int, height: int, region: str, invert: bool
) -> Optional[str]:
    meshes = []
    offset = 0

    for location in locations:
        voxels = [
            (x, width - y - 1, z)
            for x, y, z in get_location_voxels(location, region if region != "DEFAULT" else None)
        ]

        if len(voxels) == 0:
            continue

        center = [length / 2, width / 2, height / 2]
        array = make_mesh_array(voxels, length, width, height)
        verts, faces, normals = make_mesh_geometry(array, center, offset)
        mesh = make_mesh_file(verts, faces, normals, invert)

        meshes.append(mesh)
        offset = offset + len(verts)

    combined_mesh = "\n".join(meshes)

    return combined_mesh if meshes else None


def make_mesh_array(
    voxels: list[tuple[int, int, int]], length: int, width: int, height: int
) -> np.ndarray:
    # Create array.
    array = np.zeros((length, width, height), dtype=np.uint8)
    array[tuple(np.transpose(voxels))] = 7

    # Get set of zero neighbors for all voxels.
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbors = {
        (x + i, y + j, z + k)
        for x, y, z in voxels
        for i, j, k in offsets
        if array[x + i, y + j, z + k] == 0
    }

    # Remove invalid neighbors on borders.
    neighbors = {
        (x, y, z)
        for x, y, z in neighbors
        if x != 0 and y != 0 and z != 0 and x != length - 1 and y != width - 1 and z != height - 1
    }

    # Smooth array levels based on neighbor counts.
    for x, y, z in neighbors:
        array[x, y, z] = sum(array[x + i, y + j, z + k] == 7 for i, j, k in offsets) + 1

    return array


def make_mesh_geometry(
    array: np.ndarray, center: list[float], offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts, faces, normals, _ = measure.marching_cubes(array, level=3, allow_degenerate=False)

    # Center the vertices.
    verts[:, 0] = verts[:, 0] - center[0]
    verts[:, 1] = verts[:, 1] - center[1]
    verts[:, 2] = verts[:, 2] - center[2]

    # Adjust face indices.
    faces = faces + 1 + offset

    return verts, faces, normals


def make_mesh_file(
    verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, invert: bool = False
) -> str:
    mesh = ""

    for item in verts:
        mesh += f"v {item[0]} {item[1]} {item[2]}\n"

    for item in normals:
        mesh += f"vn {item[0]} {item[1]} {item[2]}\n"

    for item in faces:
        if invert:
            mesh += f"f {item[0]}//{item[0]} {item[1]}//{item[1]} {item[2]}//{item[2]}\n"
        else:
            mesh += f"f {item[2]}//{item[2]} {item[1]}//{item[1]} {item[0]}//{item[0]}\n"

    return mesh
