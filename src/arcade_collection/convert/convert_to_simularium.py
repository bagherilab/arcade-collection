import itertools
import random
from typing import Optional, Union

import numpy as np
import pandas as pd
from simulariumio import (
    DISPLAY_TYPE,
    AgentData,
    CameraData,
    DimensionData,
    DisplayData,
    MetaData,
    ModelMetaData,
    TrajectoryConverter,
    TrajectoryData,
    UnitData,
)

CAMERA_POSITIONS: dict[str, tuple[float, float, float]] = {
    "patch": (0.0, -0.5, 900),
    "potts": (10.0, 0.0, 200.0),
}

CAMERA_LOOK_AT: dict[str, tuple[float, float, float]] = {
    "patch": (0.0, -0.2, 0.0),
    "potts": (10.0, 0.0, 0.0),
}


def convert_to_simularium(
    series_key: str,
    simulation_type: str,
    data: pd.DataFrame,
    length: Union[int, float],
    width: Union[int, float],
    height: Union[int, float],
    ds: float,
    dz: float,
    dt: float,
    colors: dict[str, str],
    url: Optional[str] = None,
    jitter: float = 1.0,
) -> str:
    meta_data = get_meta_data(series_key, simulation_type, length, width, height, ds, dz)
    agent_data = get_agent_data(data)
    agent_data.display_data = get_display_data(data, colors, url, jitter)

    for index, (frame, group) in enumerate(data.groupby("frame")):
        n_agents = len(group)
        agent_data.times[index] = float(frame) * dt
        agent_data.n_agents[index] = n_agents
        agent_data.unique_ids[index][:n_agents] = range(0, n_agents)
        agent_data.types[index][:n_agents] = group["name"]
        agent_data.radii[index][:n_agents] = group["radius"]
        agent_data.positions[index][:n_agents] = group[["x", "y", "z"]]
        agent_data.n_subpoints[index][:n_agents] = group["points"].map(len)
        agent_data.viz_types[index][:n_agents] = group["points"].map(
            lambda points: 1001 if points else 1000
        )
        agent_data.subpoints[index][:n_agents] = np.array(
            list(itertools.zip_longest(*group["points"], fillvalue=0))
        ).T

    agent_data.positions[:, :, 0] = (agent_data.positions[:, :, 0] - length / 2.0) * ds
    agent_data.positions[:, :, 1] = (width / 2.0 - agent_data.positions[:, :, 1]) * ds
    agent_data.positions[:, :, 2] = (agent_data.positions[:, :, 2] - height / 2.0) * dz

    agent_data.subpoints[:, :, 0::3] = (agent_data.subpoints[:, :, 0::3]) * ds
    agent_data.subpoints[:, :, 1::3] = (-agent_data.subpoints[:, :, 1::3]) * ds
    agent_data.subpoints[:, :, 2::3] = (agent_data.subpoints[:, :, 2::3]) * dz

    return TrajectoryConverter(
        TrajectoryData(
            meta_data=meta_data,
            agent_data=agent_data,
            time_units=UnitData("hr"),
            spatial_units=UnitData("um"),
        )
    ).to_JSON()


def get_meta_data(
    series_key: str,
    simulation_type: str,
    length: Union[int, float],
    width: Union[int, float],
    height: Union[int, float],
    ds: float,
    dz: float,
) -> MetaData:
    meta_data = MetaData(
        box_size=np.array([length * ds, width * ds, height * dz]),
        camera_defaults=CameraData(
            position=np.array(CAMERA_POSITIONS[simulation_type]),
            look_at_position=np.array(CAMERA_LOOK_AT[simulation_type]),
            fov_degrees=60.0,
        ),
        trajectory_title=f"ARCADE - {series_key}",
        model_meta_data=ModelMetaData(
            title="ARCADE",
            version=simulation_type,
            description=(f"Agent-based modeling framework ARCADE for {series_key}."),
        ),
    )

    return meta_data


def get_agent_data(data: pd.DataFrame) -> AgentData:
    total_frames = len(data["frame"].unique())
    max_agents = data.groupby("frame")["name"].count().max()
    max_subpoints = data["points"].map(len).max()
    return AgentData.from_dimensions(DimensionData(total_frames, max_agents, max_subpoints))


def get_display_data(
    data: pd.DataFrame,
    colors: dict[str, str],
    url: str = "",
    jitter: float = 1.0,
) -> DisplayData:
    """
    Create map of DisplayData objects.

    Method uses the "name" and "display_type" columns in data to generate the
    DisplayData objects.

    The "name" column should be a string in one of the following forms:

    - ``(index)#(color_key)``
    - ``(group)#(color_key)#(index)``
    - ``(group)#(color_key)#(index)#(frame)``

    where ``(index)`` becomes DisplayData object name and ``(color_key)`` is
    passed to the color mapping to select the DisplayData color (optional color
    jitter may be applied).

    The "display_type" column should be a valid ``DISPLAY_TYPE``. For the
    ``DISPLAY_TYPE.OBJ`` type, a URL prefix must be used and names should be in
    the form ``(group)#(color_key)#(index)#(frame)``, which is used to generate
    the full URL formatted as: ``(url)/(frame)_(group)_(index).MESH.obj``. Note
    that ``(frame)`` is zero-padded to six digits and ``(index)`` is zero-padded
    to three digits.

    Parameters
    ----------
    data
        Simulation trajectory data.
    colors
        Color mapping.
    url
        Url prefix for meshes.
    jitter
        Jitter applied to colors.

    Returns
    -------
    :
        Map of DisplayData objects.
    """

    display_data = {}
    display_types = sorted(set(zip(data["name"], data["display_type"])))

    for name, display_type in display_types:
        if name.count("#") == 1:
            index, color_key = name.split("#")
        elif name.count("#") == 2:
            _, color_key, index = name.split("#")
        elif name.count("#") == 3:
            group, color_key, index, frame = name.split("#")

        if url != "":
            full_url = f"{url}/{int(frame):06d}_{group}_{int(index):03d}.MESH.obj"
        else:
            full_url = ""

        random.seed(index)
        alpha = jitter * (random.random() - 0.5) / 2

        display_data[name] = DisplayData(
            name=index,
            display_type=DISPLAY_TYPE[display_type],
            color=shade_color(colors[color_key], alpha),
            url=full_url,
        )

    return display_data


def shade_color(color: str, alpha: float) -> str:
    """
    Shade color by specified alpha.

    Positive values of alpha will blend the given color with white (alpha = 1.0
    returns pure white), while negative values of alpha will blend the given
    color with black (alpha = -1.0 returns pure black). An alpha = 0.0 will
    leave the color unchanged.

    Parameters
    ----------
    color
        Original color as hex string.
    alpha
        Shading value between -1 and +1.

    Returns
    -------
    :
        Shaded color as hex string.
    """

    old_color = color.replace("#", "")
    old_red, old_green, old_blue = [int(old_color[i : i + 2], 16) for i in (0, 2, 4)]
    layer_color = 0 if alpha < 0 else 255

    new_red = round(old_red + (layer_color - old_red) * abs(alpha))
    new_green = round(old_green + (layer_color - old_green) * abs(alpha))
    new_blue = round(old_blue + (layer_color - old_blue) * abs(alpha))

    return f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
