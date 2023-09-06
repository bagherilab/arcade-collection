import tarfile
from typing import Optional

import matplotlib.figure as mpl
import numpy as np
from arcade_collection.output import extract_tick_json, get_location_voxels
from prefect import task

from cell_abm_pipeline.utilities.plot import (
    add_frame_scalebar,
    add_frame_timestamp,
    make_grid_figure,
)


@task
def make_projection_frame(
    keys: list[tuple[str, int]],
    locations: dict[tuple[str, int], tarfile.TarFile],
    name: str,
    tick: int,
    scale: int,
    ds: float,
    dt: float,
    box: tuple[int, int, int],
    regions: list[str],
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    length, width, height = box

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([0, length - 1])
        ax.set_ylim([width - 1, 0])
        ax.set(aspect=1)

        ax_horz = ax.inset_axes([0, 1.005, 1, height / width], sharex=ax)
        ax_horz.set_ylim([0, height - 1])
        ax_horz.get_yaxis().set_ticks([])

        ax_vert = ax.inset_axes([1.005, 0, height / length, 1], sharey=ax)
        ax_vert.set_xlim([0, height - 1])
        ax_vert.get_xaxis().set_ticks([])

        tick_json = extract_tick_json.fn(
            locations[(key, seed)], f"{name}_{key}_{seed:04d}", tick, "LOCATIONS"
        )

        if len(regions) == 1:
            region = None if regions[0] == "DEFAULT" else regions[0]

            arr_1 = create_projection_array(tick_json, length, width, height, region)
            ax.imshow(arr_1, cmap="bone", interpolation="none", vmin=0, vmax=1)

            arr_2 = create_projection_array(tick_json, length, width, height, region, (0, 2, 1))
            ax_horz.imshow(arr_2, cmap="bone", interpolation="none", vmin=0, vmax=1)

            arr_3 = create_projection_array(tick_json, length, width, height, region, (2, 1, 0))
            ax_vert.imshow(arr_3, cmap="bone", interpolation="none", vmin=0, vmax=1)
        elif len(regions) == 2:
            region_a = None if regions[0] == "DEFAULT" else regions[0]
            region_b = None if regions[1] == "DEFAULT" else regions[1]

            arr_a1 = create_projection_array(tick_json, length, width, height, region_a)
            arr_b1 = create_projection_array(tick_json, length, width, height, region_b)
            arr_1 = join_projection_arrays(arr_a1, arr_b1)
            ax.imshow(arr_1, interpolation="none")

            arr_a2 = create_projection_array(tick_json, length, width, height, region_a, (0, 2, 1))
            arr_b2 = create_projection_array(tick_json, length, width, height, region_b, (0, 2, 1))
            arr_2 = join_projection_arrays(arr_a2, arr_b2)
            ax_horz.imshow(arr_2, interpolation="none")

            arr_a3 = create_projection_array(tick_json, length, width, height, region_a, (2, 1, 0))
            arr_b3 = create_projection_array(tick_json, length, width, height, region_b, (2, 1, 0))
            arr_3 = join_projection_arrays(arr_a3, arr_b3)
            ax_vert.imshow(arr_3, interpolation="none")

        add_frame_timestamp(ax, length, width, dt, tick, "#ffffff")
        add_frame_scalebar(ax, length, width, ds, scale, "#ffffff")

    return fig


def create_projection_array(
    locations: list,
    length: int,
    width: int,
    height: int,
    region: Optional[str] = None,
    rotate: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    array = np.zeros((length, width, height))

    for location in locations:
        voxels = get_location_voxels.fn(location, region)

        if len(voxels) == 0:
            continue

        array[tuple(np.transpose(voxels))] = location["id"]

    if rotate is not None:
        array = np.moveaxis(array, [0, 1, 2], rotate)
        length, width, height = array.shape

    borders = np.zeros((width, length))

    for i in range(length):
        for j in range(width):
            for k in range(height):
                target = array[i][j][k]

                if target != 0:
                    neighbors = [
                        1
                        for ii in [-1, 0, 1]
                        for jj in [-1, 0, 1]
                        if array[i + ii][j + jj][k] == target
                    ]
                    borders[j][i] += 9 - sum(neighbors)

    normalize = borders.max()
    borders = borders / normalize

    return borders


def join_projection_arrays(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    length, width = array_a.shape
    array = np.zeros((length, width, 3))

    array[:, :, 0] = array_a
    array[:, :, 2] = array_a

    array[:, :, 1] = array_b
    array[:, :, 2] = np.maximum(array[:, :, 2], array_b)

    return array
