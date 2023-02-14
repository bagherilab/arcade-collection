import tarfile

import numpy as np
from prefect import task
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

from arcade_collection.output.extract_tick_json import extract_tick_json


@task
def convert_to_simularium(
    series_key: str,
    cells_data_tar: tarfile.TarFile,
    locations_data_tar: tarfile.TarFile,
    frame_spec: tuple[int, int, int],
    box: tuple[int, int, int],
    ds: float,
    dt: float,
    phase_colors: dict[str, str],
) -> str:
    length, width, height = box
    frames = list(np.arange(*frame_spec))

    meta_data = get_meta_data(series_key, length, width, height, ds)
    agent_data = get_agent_data(series_key, cells_data_tar, frames)
    agent_data.display_data = get_display_data(phase_colors)

    for index, frame in enumerate(frames):
        agent_data.times[index] = float(frame) * dt
        convert_cells_tar(agent_data, cells_data_tar, series_key, frame, index)
        convert_locations_tar(
            agent_data, locations_data_tar, series_key, frame, index, length, width, height, ds
        )

    return TrajectoryConverter(
        TrajectoryData(
            meta_data=meta_data,
            agent_data=agent_data,
            time_units=UnitData("hr"),
            spatial_units=UnitData("um"),
        )
    ).to_JSON()


def get_meta_data(series_key: str, length: int, width: int, height: int, ds: float) -> MetaData:
    meta_data = MetaData(
        box_size=np.array([length * ds, width * ds, height * ds]),
        camera_defaults=CameraData(
            position=np.array([10.0, 0.0, 200.0]),
            look_at_position=np.array([10.0, 0.0, 0.0]),
            fov_degrees=60.0,
        ),
        trajectory_title=f"ARCADE - {series_key}",
        model_meta_data=ModelMetaData(
            title="ARCADE",
            version="3.0",
            description=(f"Agent-based modeling framework ARCADE for {series_key}."),
        ),
    )

    return meta_data


def get_agent_data(series_key: str, cells_tar: tarfile.TarFile, frames: list[int]) -> AgentData:
    total_frames = len(frames)

    max_agents = 0
    for frame in frames:
        cells = extract_tick_json.fn(cells_tar, series_key, frame, "CELLS")
        max_agents = max(max_agents, len(cells))

    return AgentData.from_dimensions(DimensionData(total_frames, max_agents))


def get_display_data(phase_colors: dict[str, str]) -> DisplayData:
    display_data = {}

    for phase, color in phase_colors.items():
        display_data[phase] = DisplayData(name=phase, color=color, display_type=DISPLAY_TYPE.SPHERE)

    return display_data


def convert_cells_tar(
    data: AgentData,
    tar: tarfile.TarFile,
    series_key: str,
    frame: int,
    index: int,
) -> None:
    cells = extract_tick_json.fn(tar, series_key, frame, "CELLS")
    data.n_agents[index] = len(cells)

    for i, cell in enumerate(cells):
        data.unique_ids[index][i] = cell["id"]
        data.types[index].append(cell["phase"])
        data.radii[index][i] = (cell["voxels"] ** (1.0 / 3)) / 1.5


def convert_locations_tar(
    data: AgentData,
    tar: tarfile.TarFile,
    series_key: str,
    frame: int,
    index: int,
    length: int,
    width: int,
    height: int,
    ds: float,
) -> None:
    locations = extract_tick_json.fn(tar, series_key, frame, "LOCATIONS")

    for i, location in enumerate(locations):
        data.positions[index][i] = np.array(
            [
                (location["center"][0] - length / 2) * ds,
                (width / 2 - location["center"][1]) * ds,
                (location["center"][2] - height / 2) * ds,
            ]
        )
