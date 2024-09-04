import tarfile

import numpy as np
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels

ROTATIONS = {
    "top": (0, 1, 2),
    "side1": (0, 2, 1),
    "side2": (2, 1, 0),
}


def convert_to_contours(
    series_key: str,
    data_tar: tarfile.TarFile,
    frame: int,
    regions: list[str],
    box: tuple[int, int, int],
    indices: dict[str, list[int]],
) -> dict[str, dict[str, dict[int, list]]]:
    locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

    contours: dict[str, dict[str, dict[int, list]]] = {
        region: {view: {} for view in indices.keys()} for region in regions
    }

    for location in locations:
        for region in regions:
            array = np.zeros(box)
            voxels = get_location_voxels(location, region)

            if len(voxels) == 0:
                continue

            array[tuple(np.transpose(voxels))] = 1

            for view in indices.keys():
                array_rotated = np.moveaxis(array, [0, 1, 2], ROTATIONS[view])

                for index in indices[view]:
                    array_slice = array_rotated[:, :, index]

                    if np.sum(array_slice) == 0:
                        continue

                    if index not in contours[region][view]:
                        contours[region][view][index] = []

                    contours[region][view][index].extend(measure.find_contours(array_slice))

    return contours
