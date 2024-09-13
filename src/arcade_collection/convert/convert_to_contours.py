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
"""Axis rotations for different contour views."""


def convert_to_contours(
    series_key: str,
    locations_tar: tarfile.TarFile,
    frame: int,
    regions: list[str],
    box: tuple[int, int, int],
    indices: dict[str, list[int]],
) -> dict[str, dict[str, dict[int, list]]]:
    """
    Convert data to iso-valued contours.

    Contours are calculated using "marching squares" method. Note that these
    contours follow the iso-values, which means that a single "square" voxel
    will produce a diamond-shaped contour. For the exact outline of a set of
    voxels, consider using ``extract_voxel_contours`` from the
    ``abm_shape_collection`` package.

    Parameters
    ----------
    series_key
        Simulation series key.
    locations_tar
        Archive of location data.
    frame : int
        _description_
    regions
        List of regions.
    box
        Size of bounding box.
    indices
        Map of view to slice indices.

    Returns
    -------
    :
        Map of region, view, and index to contours.
    """

    locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

    contours: dict[str, dict[str, dict[int, list]]] = {
        region: {view: {} for view in indices} for region in regions
    }

    for location in locations:
        for region in regions:
            array = np.zeros(box)
            voxels = get_location_voxels(location, region if region != "DEFAULT" else None)

            if len(voxels) == 0:
                continue

            array[tuple(np.transpose(voxels))] = 1

            for view in indices:
                array_rotated = np.moveaxis(array, [0, 1, 2], ROTATIONS[view])

                for index in indices[view]:
                    array_slice = array_rotated[:, :, index]

                    if np.sum(array_slice) == 0:
                        continue

                    if index not in contours[region][view]:
                        contours[region][view][index] = []

                    array_contours = [
                        contour.tolist() for contour in measure.find_contours(array_slice)
                    ]
                    contours[region][view][index].extend(array_contours)

    return contours
