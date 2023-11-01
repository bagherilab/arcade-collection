import tarfile
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import measure

from arcade_collection.output.extract_tick_json import extract_tick_json
from arcade_collection.output.get_location_voxels import get_location_voxels


def convert_to_projection(
    series_key: str,
    data_tar: tarfile.TarFile,
    frame: int,
    regions: list[str],
    box: tuple[int, int, int],
    ds: float,
    dt: float,
    scale: int,
    colors: dict[str, str],
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    length, width, height = box

    ax = fig.add_subplot()

    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim([0, length - 1])
    ax.set_ylim([width - 1, 0])
    ax.set_box_aspect(1)

    ax_horz = ax.inset_axes([0, 1.005, 1, height / width], sharex=ax)
    ax_horz.set_ylim([0, height - 1])
    ax_horz.get_yaxis().set_ticks([])

    ax_vert = ax.inset_axes([1.005, 0, height / length, 1], sharey=ax)
    ax_vert.set_xlim([0, height - 1])
    ax_vert.get_xaxis().set_ticks([])

    ax.set_facecolor("#000")
    ax_horz.set_facecolor("#000")
    ax_vert.set_facecolor("#000")

    locations = extract_tick_json(data_tar, series_key, frame, "LOCATIONS")

    for region in regions:
        color = colors[region]

        for location in locations:
            for contour in get_array_contours(location, length, width, height, region):
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

            for contour in get_array_contours(location, length, width, height, region, (0, 2, 1)):
                ax_horz.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

            for contour in get_array_contours(location, length, width, height, region, (2, 1, 0)):
                ax_vert.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

    add_frame_timestamp(ax, length, width, dt, frame, "#ffffff")
    add_frame_scalebar(ax, length, width, ds, scale, "#ffffff")

    return fig


def get_array_contours(
    location: dict,
    length: int,
    width: int,
    height: int,
    region: Optional[str] = None,
    rotate: Optional[tuple[int, int, int]] = None,
) -> list[np.ndarray]:
    array = np.zeros((length, width, height))
    voxels = get_location_voxels(location, region)

    if len(voxels) == 0:
        return []

    array[tuple(np.transpose(voxels))] = 1

    if rotate is not None:
        array = np.moveaxis(array, [0, 1, 2], rotate)
        length, width, height = array.shape

    contours: list[np.ndarray] = []

    for z in range(1, height):
        array_slice = array[:, :, z]

        if np.sum(array_slice) == 0:
            continue

        contours = contours + measure.find_contours(array_slice)

    return contours


def add_frame_timestamp(
    ax: mpl.axes.Axes, length: int, width: int, dt: float, frame: int, color: str
) -> None:
    hours, minutes = divmod(round(frame * dt, 2), 1)
    timestamp = f"{int(hours):02d}H:{round(minutes*60):02d}M"

    ax.text(
        0.03 * length,
        0.96 * width,
        timestamp,
        fontfamily="monospace",
        fontsize=20,
        color=color,
        fontweight="bold",
    )


def add_frame_scalebar(
    ax: mpl.axes.Axes, length: int, width: int, ds: float, scale: int, color: str
) -> None:
    scalebar = scale / ds

    ax.add_patch(
        Rectangle(
            (0.95 * length - scalebar, 0.94 * width),
            scalebar,
            0.01 * width,
            snap=True,
            color=color,
        )
    )

    ax.text(
        0.95 * length - scalebar / 2,
        0.975 * width,
        f"{scale} $\\mu$m",
        fontsize=10,
        color=color,
        horizontalalignment="center",
    )
