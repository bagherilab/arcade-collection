import tarfile

import numpy as np
from prefect import task

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


@task
def convert_to_images(
    series_key: str,
    data: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    regions: list[str],
    box: tuple[int, int, int],
    chunk_size: int,
    binary: bool,
) -> list[tuple[int, int, np.ndarray]]:
    length, width, height = box
    frames = list(np.arange(*frame_spec))
    array = np.zeros((len(frames), len(regions), height, width, length), "uint16")

    for index, frame in enumerate(frames):
        locations = extract_tick_json.fn(data, series_key, frame, "LOCATIONS")

        for location in locations:
            location_id = location["id"]

            for channel, region in enumerate(regions):
                voxels = [
                    (z, y, x)
                    for x, y, z in get_location_voxels.fn(
                        location, region if region != "DEFAULT" else None
                    )
                ]
                array[index, channel][tuple(np.transpose(voxels))] = 1 if binary else location_id

    return split_array_chunks(array, chunk_size)


def split_array_chunks(array: np.ndarray, chunk_size: int) -> list[tuple[int, int, np.ndarray]]:
    chunks = []
    length = array.shape[4]
    width = array.shape[3]

    # Calculate chunk splits.
    length_section = (
        [0, chunk_size + 1] + (int(length / chunk_size) - 2) * [chunk_size] + [chunk_size + 1]
    )
    length_splits = np.array(length_section, dtype=np.int32).cumsum()
    width_section = (
        [0, chunk_size + 1] + (int(width / chunk_size) - 2) * [chunk_size] + [chunk_size + 1]
    )
    width_splits = np.array(width_section, dtype=np.int32).cumsum()

    # Iterate through each chunk split.
    for i in range(len(length_splits) - 1):
        length_start = length_splits[i]
        length_end = length_splits[i + 1]

        for j in range(len(width_splits) - 1):
            width_start = width_splits[j]
            width_end = width_splits[j + 1]

            # Extract chunk from full contents.
            chunk = array[:, :, :, length_start:length_end, width_start:width_end]

            if np.sum(chunk) != 0:
                chunks.append((i, j, chunk))

    return chunks
