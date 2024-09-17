import tarfile
from typing import Optional, Union

import numpy as np
import pandas as pd
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels

MAX_ARRAY_LEVEL = 7
"""Maximum array level for conversion to meshes."""


def convert_to_meshes(
    series_key: str,
    locations_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    invert: Union[bool, dict[str, bool]] = False,
    group_size: Optional[int] = None,
    categories: Optional[pd.DataFrame] = None,
) -> list[tuple[int, int, str, str]]:
    """
    Convert data to mesh OBJ contents.

    Parameters
    ----------
    series_key
        Simulation series key.
    locations_tar
        Archive of location data.
    frame_spec
        Specification for mesh frames.
    regions
        List of regions.
    box
        Size of bounding box.
    invert
        True to invert the order of faces, False otherwise.
    group_size
        Number of objects in each group (if grouping meshes).
    categories
        Simulation data containing ID, FRAME, and CATEGORY.

    Returns
    -------
    :
        List of mesh frames, indices, regions, and OBJ contents.
    """

    frames = list(np.arange(*frame_spec))
    meshes = []

    length, width, height = box

    if group_size is not None:
        groups = make_mesh_groups(categories, frames, group_size)
    else:
        groups = None

    for frame in frames:
        locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

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
    """
    Group objects based on group size and categories.

    Parameters
    ----------
    categories
        Simulation data containing ID, FRAME, and CATEGORY.
    frames
        List of frames.
    group_size
        Number of objects in each group.

    Returns
    -------
    :
        Map of frame to map of index to location ids.
    """

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
    """
    Create mesh containing a single object.

    Parameters
    ----------
    location
        Location object.
    length
        Bounding box length.
    width
        Bounding box width.
    height
        Bounding box height.
    region
        Region name.
    invert
        True to invert the order of faces, False otherwise.

    Returns
    -------
    :
        Single mesh OBJ file contents.
    """

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
    """
    Create mesh containing multiple objects.

    Parameters
    ----------
    locations
        List of location objects.
    length
        Bounding box length.
    width
        Bounding box width.
    height
        Bounding box height.
    region
        Region name.
    invert
        True to invert the order of faces, False otherwise.

    Returns
    -------
    :
        Combined mesh OBJ file contents.
    """

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
    """
    Generate array from list of voxels.

    Given voxel locations are set to the max array level. The array is smoothed
    such that all other locations are set to the number of max-level neighbors.

    Parameters
    ----------
    voxels
        List of voxels representing object.
    length
        Bounding box length.
    width
        Bounding box width.
    height
        Bounding box height.

    Returns
    -------
    :
        Array representing object.
    """

    # Create array.
    array = np.zeros((length, width, height), dtype=np.uint8)
    array[tuple(np.transpose(voxels))] = MAX_ARRAY_LEVEL

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
        array[x, y, z] = (
            sum(array[x + i, y + j, z + k] == MAX_ARRAY_LEVEL for i, j, k in offsets) + 1
        )

    return array


def make_mesh_geometry(
    array: np.ndarray, center: list[float], offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mesh from array.

    Parameters
    ----------
    array
        Array representing object.
    center
        Coordinate of object center.
    offset
        Offset for face indices.

    Returns
    -------
    :
        Arrays of mesh vertices, faces, and normals.
    """

    level = int(MAX_ARRAY_LEVEL / 2)
    verts, faces, normals, _ = measure.marching_cubes(array, level=level, allow_degenerate=False)

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
    """
    Create mesh OBJ file contents from marching cubes output.

    If

    Parameters
    ----------
    verts
        Array of mesh vertices.
    faces
        Array of mesh faces.
    normals
        Array of mesh normals.
    invert
        True to invert the order of faces, False otherwise.

    Returns
    -------
    :
        Mesh OBJ file.
    """

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
