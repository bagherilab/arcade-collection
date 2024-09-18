import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arcade_collection.convert.convert_to_contours import convert_to_contours


def convert_to_projection(
    series_key: str,
    locations_tar: tarfile.TarFile,
    frame: int,
    regions: list[str],
    box: tuple[int, int, int],
    ds: float,
    dt: float,
    scale: int,
    colors: dict[str, str],
) -> mpl.figure.Figure:
    """
    Convert data to projection figure.

    Parameters
    ----------
    series_key
        Simulation series key.
    locations_tar
        Archive of location data.
    frame
        Frame number.
    regions
        List of regions.
    box
        Size of bounding box.
    ds
        Spatial scaling in um/voxel.
    dt
        Temporal scaling in hours/tick.
    scale
        Size of scale bar (in um).
    colors
        Map of region to colors.

    Returns
    -------
    :
        Projection figure.
    """

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

    indices = {
        "top": list(range(1, height)),
        "side1": list(range(1, width)),
        "side2": list(range(1, length)),
    }
    contours = convert_to_contours(series_key, locations_tar, frame, regions, box, indices)

    for region in regions:
        color = colors[region]

        for region_top_contours in contours[region]["top"].values():
            for contour in region_top_contours:
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

        for region_side1_contours in contours[region]["side1"].values():
            for contour in region_side1_contours:
                ax_horz.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

        for region_side2_contours in contours[region]["side2"].values():
            for contour in region_side2_contours:
                ax_vert.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color, alpha=0.5)

    add_frame_timestamp(ax, length, width, dt, frame, "#ffffff")
    add_frame_scalebar(ax, length, width, ds, scale, "#ffffff")

    return fig


def add_frame_timestamp(
    ax: mpl.axes.Axes, length: int, width: int, dt: float, frame: int, color: str
) -> None:
    """
    Add a frame timestamp to figure axes.

    Parameters
    ----------
    ax
        Axes object.
    length
        Length of bounding box.
    width
        Width of bounding box.
    dt
        Temporal scaling in hours/tick.
    frame
        Frame number.
    color
        Timestamp color.
    """

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
    """
    Add a frame scalebar to figure axes.

    Parameters
    ----------
    ax
        Axes object.
    length
        Length of bounding box.
    width
        Width of bounding box.
    ds
        Spatial scaling in um/voxel.
    scale
        Size of scale bar (in um).
    color
        Scalebar color.
    """

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
