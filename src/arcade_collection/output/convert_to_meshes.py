import tarfile

import numpy as np
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def convert_to_meshes(
    series_key: str, data_tar: tarfile.TarFile, frame_spec: tuple[int, int, int], regions: list[str]
) -> list[tuple[int, int, str, str]]:
    frames = list(np.arange(*frame_spec))
    meshes = []

    for frame in frames:
        locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

        for location in locations:
            location_id = location["id"]

            for region in regions:
                voxels = [
                    (z, y, x)
                    for x, y, z in get_location_voxels(
                        location, region if region != "DEFAULT" else None
                    )
                ]

                if len(voxels) == 0:
                    continue

                array = make_mesh_array(voxels)
                mesh = make_mesh_object(array)

                meshes.append((frame, location_id, region, mesh))

    return meshes


def make_mesh_array(voxels: list[tuple[int, int, int]]) -> np.ndarray:
    mins = np.min(voxels, axis=0)
    maxs = np.max(voxels, axis=0)
    height, width, length = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint8)

    voxels_transposed = [voxel - mins + 1 for voxel in voxels]
    array[tuple(np.transpose(voxels_transposed))] = 7

    # Get set of zero neighbors for all voxels.
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbors = {
        (z + i, y + j, x + k)
        for z, y, x in voxels_transposed
        for k, j, i in offsets
        if array[z + k, y + j, x + i] == 0
    }

    # Remove invalid neighbors on borders.
    neighbors = {
        (z, y, x)
        for z, y, x in neighbors
        if x != 0 and y != 0 and z != 0 and x != length - 1 and y != width - 1 and z != height - 1
    }

    # Smooth array levels based on neighbor counts.
    for z, y, x in neighbors:
        array[z, y, x] = sum(array[z + k, y + j, x + i] == 7 for k, j, i in offsets) + 1

    return array


def make_mesh_object(array: np.ndarray) -> str:
    verts, faces, normals, _ = measure.marching_cubes(array, level=3, allow_degenerate=False)
    mesh = ""
    faces = faces + 1

    for item in verts:
        mesh += f"v {item[0]} {item[1]} {item[2]}\n"

    for item in normals:
        mesh += f"vn {item[0]} {item[1]} {item[2]}\n"

    for item in faces:
        mesh += f"f {item[2]}//{item[2]} {item[1]}//{item[1]} {item[0]}//{item[0]}\n"

    return mesh
