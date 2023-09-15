import tarfile

import numpy as np
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def convert_to_meshes(
    series_key: str,
    data_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    invert: bool = False,
) -> list[tuple[int, int, str, str]]:
    frames = list(np.arange(*frame_spec))
    meshes = []

    length, width, height = box

    for frame in frames:
        locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

        for location in locations:
            location_id = location["id"]

            for region in regions:
                voxels = [
                    (x, width - y - 1, z)
                    for x, y, z in get_location_voxels(
                        location, region if region != "DEFAULT" else None
                    )
                ]

                if len(voxels) == 0:
                    continue

                center = list(np.array(voxels).mean(axis=0))
                array = make_mesh_array(voxels, length, width, height)
                mesh = make_mesh_object(array, center, invert)

                meshes.append((frame, location_id, region, mesh))

    return meshes


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


def make_mesh_object(array: np.ndarray, center: list[float], invert: bool = False) -> str:
    verts, faces, normals, _ = measure.marching_cubes(array, level=3, allow_degenerate=False)
    mesh = ""
    faces = faces + 1

    for item in verts:
        mesh += f"v {item[0] - center[0]} {item[1] - center[1]} {item[2] - center[2]}\n"

    for item in normals:
        mesh += f"vn {item[0]} {item[1]} {item[2]}\n"

    for item in faces:
        if invert:
            mesh += f"f {item[0]}//{item[0]} {item[1]}//{item[1]} {item[2]}//{item[2]}\n"
        else:
            mesh += f"f {item[2]}//{item[2]} {item[1]}//{item[1]} {item[0]}//{item[0]}\n"

    return mesh
