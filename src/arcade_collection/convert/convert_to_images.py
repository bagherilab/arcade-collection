from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels

if TYPE_CHECKING:
    import tarfile


class ImageType(Enum):
    """Image conversion types."""

    FULL = (False, False, False)
    """Image with TCZYX dimensions."""

    FULL_BINARY = (True, False, False)
    """Binary image with TCZYX dimensions."""

    FULL_BY_FRAME = (False, True, False)
    """Image with TCZYX dimensions separated by frame."""

    FULL_BINARY_BY_FRAME = (True, True, False)
    """Binary image with TCZYX dimensions separated by frame."""

    FLAT_RGBA_BY_FRAME = (False, True, True)
    """RGBA array flattened to YX dimensions separated by frame."""

    FLAT_BINARY_BY_FRAME = (True, True, True)
    """Binary array flattened to YX dimensions separated by frame."""


def convert_to_images(
    series_key: str,
    locations_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    chunk_size: int,
    image_type: ImageType,
) -> list[tuple[int, int, np.ndarray, int | None]]:
    """
    Convert data to image arrays.

    Images are extracted from lists of voxels. The initial converted image has
    dimensions in TCZYX order, such that T encodes the specified frames and C
    encodes the regions. The initial converted image is then further processed
    based on selected image type.

    Parameters
    ----------
    series_key
        Simulation series key.
    locations_tar
        Archive of location data.
    frame_spec
        Specification for image frames.
    regions
        List of region channels.
    box
        Size of bounding box.
    chunk_size
        Size of each image chunk.
    image_type
        Image conversion type.

    Returns
    -------
    :
        List of image chunks, chunk indices, and frames.
    """

    binary, separate, _ = image_type.value
    length, width, height = box
    frames = list(np.arange(*frame_spec))
    raw_array = np.zeros((len(frames), len(regions), height, width, length), "uint16")

    for index, frame in enumerate(frames):
        locations = extract_tick_json(locations_tar, series_key, frame, "LOCATIONS")

        for location in locations:
            value = 1 if binary else location["id"]

            for channel, region in enumerate(regions):
                voxels = [
                    (z, y, x)
                    for x, y, z in get_location_voxels(
                        location, region if region != "DEFAULT" else None
                    )
                ]

                if len(voxels) == 0:
                    continue

                raw_array[index, channel][tuple(np.transpose(voxels))] = value

    # Remove 1 pixel border.
    array = raw_array[:, :, 1:-1, 1:-1, 1:-1].copy()

    if separate:
        chunks = [
            (i, j, flatten_array_chunk(chunk, image_type), frame)
            for index, frame in enumerate(frames)
            for i, j, chunk in split_array_chunks(array[[index], :, :, :, :], chunk_size)
        ]
    else:
        chunks = [(i, j, chunk, None) for i, j, chunk in split_array_chunks(array, chunk_size)]

    return chunks


def split_array_chunks(array: np.ndarray, chunk_size: int) -> list[tuple[int, int, np.ndarray]]:
    """
    Split arrays into smaller chunks.

    Parameters
    ----------
    array
        Image array (dimensions in TCZYX order).
    chunk_size
        Size of each image chunk.

    Returns
    -------
    :
        List of array chunks and their relative indices.
    """

    chunks = []
    length = array.shape[4]
    width = array.shape[3]

    # Calculate chunk splits.
    length_section = [0] + (int(length / chunk_size)) * [chunk_size]
    length_splits = np.array(length_section, dtype=np.int32).cumsum()
    width_section = [0] + (int(width / chunk_size)) * [chunk_size]
    width_splits = np.array(width_section, dtype=np.int32).cumsum()

    # Iterate through each chunk split.
    for i in range(len(length_splits) - 1):
        length_start = length_splits[i]
        length_end = length_splits[i + 1]

        for j in range(len(width_splits) - 1):
            width_start = width_splits[j]
            width_end = width_splits[j + 1]

            # Extract chunk from full contents.
            chunk = np.copy(array[:, :, :, length_start:length_end, width_start:width_end])

            if np.sum(chunk) != 0:
                chunks.append((i, j, chunk))

    return chunks


def flatten_array_chunk(array: np.ndarray, image_type: ImageType) -> np.ndarray:
    """
    Flatten array chunk along z axis.

    When flattening to an RGBA array, each object is encoded as a unique color
    such that the object ID = R + G*256 + B*256*256 - 1 and background pixels
    are black (R = 0, G = 0, B = 0).

    Parameters
    ----------
    array
        Image array (dimensions in TCZYX order).
    image_type
        Image conversion type.

    Returns
    -------
    :
        Flattened image array.
    """

    array_flat = array[0, 0, :, :, :].max(axis=0)

    if image_type == ImageType.FLAT_RGBA_BY_FRAME:
        array_rgba = np.zeros((*array_flat.shape, 4), dtype=np.uint8)
        array_rgba[:, :, 0] = (array_flat & 0x000000FF) >> 0
        array_rgba[:, :, 1] = (array_flat & 0x0000FF00) >> 8
        array_rgba[:, :, 2] = (array_flat & 0x00FF0000) >> 16
        array_rgba[:, :, 3] = 255  # (array_flat & 0x00FF0000) >> 24
        return array_rgba

    if image_type == ImageType.FLAT_BINARY_BY_FRAME:
        return array_flat

    return array
