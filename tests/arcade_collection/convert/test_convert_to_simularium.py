import json
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from simulariumio import DISPLAY_TYPE
from simulariumio.constants import DEFAULT_CAMERA_SETTINGS, VALUES_PER_3D_POINT, VIZ_TYPE

from arcade_collection.convert.convert_to_simularium import (
    CAMERA_LOOK_AT,
    CAMERA_POSITIONS,
    convert_to_simularium,
    get_agent_data,
    get_display_data,
    get_meta_data,
    shade_color,
)


class TestConvertToSimularium(unittest.TestCase):
    def test_convert_to_simularium(self):
        names = ["X#A#1", "X#B#2", "X#A", "X#A#3#5", "X#A#4", "X#A#5#10", "X#B#6"]
        display_types = ["FIBER", "FIBER", "FIBER", "OBJ", "SPHERE", "OBJ", "SPHERE"]

        series_key = "SERIES_KEY"
        simulation_type = "simulation_type"
        data = pd.DataFrame(
            {
                "frame": [0, 0, 5, 5, 5, 10, 10],
                "name": names,
                "points": [[0, 1, 2], [3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5], [], [], [], []],
                "display": display_types,
                "radius": [1, 2, 3, 4, 5, 6, 7],
                "x": [8, 9, 10, 11, 12, 13, 14],
                "y": [15, 16, 17, 18, 19, 20, 21],
                "z": [22, 23, 24, 25, 26, 27, 28],
            }
        )
        length = 10
        width = 20
        height = 30
        ds = (2, 3, 4)
        dt = 5
        colors = {"A": "#ff0000", "B": "#0000ff"}
        url = "https://url"
        jitter = 0.0

        total_steps = 3
        time_interval = 5
        time_step_size = dt * time_interval
        size = {"x": length * ds[0], "y": width * ds[1], "z": height * ds[2]}
        camera_position = {
            "x": DEFAULT_CAMERA_SETTINGS.CAMERA_POSITION[0],
            "y": DEFAULT_CAMERA_SETTINGS.CAMERA_POSITION[1],
            "z": DEFAULT_CAMERA_SETTINGS.CAMERA_POSITION[2],
        }
        camera_look_at = {
            "x": DEFAULT_CAMERA_SETTINGS.LOOK_AT_POSITION[0],
            "y": DEFAULT_CAMERA_SETTINGS.LOOK_AT_POSITION[1],
            "z": DEFAULT_CAMERA_SETTINGS.LOOK_AT_POSITION[2],
        }
        type_mapping_entries = [
            (names[0], display_types[0], colors["A"], None),
            (names[1], display_types[1], colors["B"], None),
            (names[2], display_types[2], colors["A"], None),
            (names[3], display_types[3], colors["A"], f"{url}/000005_X_003.MESH.obj"),
            (names[4], display_types[4], colors["A"], None),
            (names[5], display_types[5], colors["A"], f"{url}/000010_X_005.MESH.obj"),
            (names[6], display_types[6], colors["B"], None),
        ]
        type_mapping = {
            str(i): (
                {
                    "name": name,
                    "geometry": {"displayType": display_type, "url": url, "color": color},
                }
                if url is not None
                else {
                    "name": name,
                    "geometry": {"displayType": display_type, "color": color},
                }
            )
            for i, (name, display_type, color, url) in enumerate(type_mapping_entries)
        }
        title = f"ARCADE - {series_key}"
        model_info = {
            "title": "ARCADE",
            "version": simulation_type,
            "description": f"Agent-based modeling framework ARCADE for {series_key}.",
        }

        def make_bundle_data(viz_type, object_id, display_index, x0, y0, z0, radius, subpoints):
            rotation = (0, 0, 0)
            x = x0 * ds[0] - length * ds[0] / 2
            y = width * ds[1] / 2 - y0 * ds[1]
            z = z0 * ds[2] - height * ds[2] / 2

            subpoints_array = np.reshape(subpoints, (-1, 3))
            subpoints_array[:, 0] *= ds[0]
            subpoints_array[:, 1] *= -ds[1]
            subpoints_array[:, 2] *= ds[2]

            return [
                viz_type,
                object_id,
                display_index,
                x,
                y,
                z,
                *rotation,
                radius,
                len(subpoints),
                *subpoints_array.ravel().tolist(),
            ]

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(VIZ_TYPE.FIBER, 0, 0, 8, 15, 22, 1, [0, 1, 2]),
                    *make_bundle_data(VIZ_TYPE.FIBER, 1, 1, 9, 16, 23, 2, [3, 4, 5, 6, 7, 8]),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(VIZ_TYPE.FIBER, 0, 2, 10, 17, 24, 3, [0, 1, 2, 3, 4, 5]),
                    *make_bundle_data(VIZ_TYPE.DEFAULT, 1, 3, 11, 18, 25, 4, []),
                    *make_bundle_data(VIZ_TYPE.DEFAULT, 2, 4, 12, 19, 26, 5, []),
                ],
            },
            {
                "frameNumber": 2,
                "time": 2.0 * time_step_size,
                "data": [
                    *make_bundle_data(VIZ_TYPE.DEFAULT, 0, 5, 13, 20, 27, 6, []),
                    *make_bundle_data(VIZ_TYPE.DEFAULT, 1, 6, 14, 21, 28, 7, []),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium(
                series_key,
                simulation_type,
                data,
                length,
                width,
                height,
                ds,
                dt,
                colors,
                url,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(camera_position, trajectory_info["cameraDefault"]["position"])
        self.assertDictEqual(camera_look_at, trajectory_info["cameraDefault"]["lookAtPosition"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertEqual(title, trajectory_info["trajectoryTitle"])
        self.assertDictEqual(model_info, trajectory_info["modelInfo"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    def test_get_meta_data_simulation_type_with_defaults(self):
        series_key = "SERIES_KEY"
        length = 10
        width = 20
        height = 30
        dx = 2
        dy = 3
        dz = 4

        parameters = ["patch", "potts"]

        expected_box_size = np.array([length * dx, width * dy, height * dz])
        expected_trajectory_title = f"ARCADE - {series_key}"
        expected_title = "ARCADE"
        expected_description = f"Agent-based modeling framework ARCADE for {series_key}."

        for simulation_type in parameters:
            with self.subTest(simulation_type=simulation_type):
                expected_version = simulation_type
                expected_position = np.array(CAMERA_POSITIONS[simulation_type])
                expected_look_at = np.array(CAMERA_LOOK_AT[simulation_type])

                meta_data = get_meta_data(
                    series_key, simulation_type, length, width, height, dx, dy, dz
                )

                self.assertTrue((expected_box_size == meta_data.box_size).all())
                self.assertTrue((expected_position == meta_data.camera_defaults.position).all())
                self.assertTrue(
                    (expected_look_at == meta_data.camera_defaults.look_at_position).all()
                )
                self.assertEqual(expected_trajectory_title, meta_data.trajectory_title)
                self.assertEqual(expected_title, meta_data.model_meta_data.title)
                self.assertEqual(expected_version, meta_data.model_meta_data.version)
                self.assertEqual(expected_description, meta_data.model_meta_data.description)

    def test_get_meta_data_simulation_type_without_defaults(self):
        series_key = "SERIES_KEY"
        simulation_type = "simulation_type"
        length = 10
        width = 20
        height = 30
        dx = 2
        dy = 3
        dz = 4

        expected_box_size = np.array([length * dx, width * dy, height * dz])
        expected_trajectory_title = f"ARCADE - {series_key}"
        expected_title = "ARCADE"
        expected_description = f"Agent-based modeling framework ARCADE for {series_key}."
        expected_version = simulation_type
        expected_position = DEFAULT_CAMERA_SETTINGS.CAMERA_POSITION
        expected_look_at = DEFAULT_CAMERA_SETTINGS.LOOK_AT_POSITION

        meta_data = get_meta_data(series_key, simulation_type, length, width, height, dx, dy, dz)

        self.assertTrue((expected_box_size == meta_data.box_size).all())
        self.assertTrue((expected_position == meta_data.camera_defaults.position).all())
        self.assertTrue((expected_look_at == meta_data.camera_defaults.look_at_position).all())
        self.assertEqual(expected_trajectory_title, meta_data.trajectory_title)
        self.assertEqual(expected_title, meta_data.model_meta_data.title)
        self.assertEqual(expected_version, meta_data.model_meta_data.version)
        self.assertEqual(expected_description, meta_data.model_meta_data.description)

    def test_get_agent_data_no_subpoints(self):
        data = pd.DataFrame(
            {
                "frame": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                "name": ["A", "B", "C", "D", "A", "B", "C", "A", "B", "C", "D", "E"],
                "points": [[], [], [], [], [], [], [], [], [], [], [], []],
            }
        )

        total_steps = 3
        max_agents = 5
        max_subpoints = 0

        agent_data = get_agent_data(data)

        self.assertEqual((total_steps,), agent_data.times.shape)
        self.assertEqual((total_steps,), agent_data.n_agents.shape)
        self.assertEqual((total_steps, max_agents), agent_data.viz_types.shape)
        self.assertEqual((total_steps, max_agents), agent_data.unique_ids.shape)
        self.assertEqual(total_steps, len(agent_data.types))
        self.assertEqual((total_steps, max_agents, VALUES_PER_3D_POINT), agent_data.positions.shape)
        self.assertEqual((total_steps, max_agents), agent_data.radii.shape)
        self.assertEqual((total_steps, max_agents, VALUES_PER_3D_POINT), agent_data.rotations.shape)
        self.assertEqual((total_steps, max_agents), agent_data.n_subpoints.shape)
        self.assertEqual((total_steps, max_agents, max_subpoints), agent_data.subpoints.shape)

    def test_get_agent_data_with_subpoints(self):
        data = pd.DataFrame(
            {
                "frame": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                "name": ["A", "B", "C", "D", "A", "B", "C", "A", "B", "C", "D", "E"],
                "points": [[0, 1, 2], [], [], [], [], [0, 1, 2, 3, 4, 5], [], [], [], [], [], []],
            }
        )

        total_steps = 3
        max_agents = 5
        max_subpoints = 6

        agent_data = get_agent_data(data)

        self.assertEqual((total_steps,), agent_data.times.shape)
        self.assertEqual((total_steps,), agent_data.n_agents.shape)
        self.assertEqual((total_steps, max_agents), agent_data.viz_types.shape)
        self.assertEqual((total_steps, max_agents), agent_data.unique_ids.shape)
        self.assertEqual(total_steps, len(agent_data.types))
        self.assertEqual((total_steps, max_agents, VALUES_PER_3D_POINT), agent_data.positions.shape)
        self.assertEqual((total_steps, max_agents), agent_data.radii.shape)
        self.assertEqual((total_steps, max_agents, VALUES_PER_3D_POINT), agent_data.rotations.shape)
        self.assertEqual((total_steps, max_agents), agent_data.n_subpoints.shape)
        self.assertEqual((total_steps, max_agents, max_subpoints), agent_data.subpoints.shape)

    def test_get_display_data_no_url(self):
        data = pd.DataFrame(
            {
                "name": ["GROUP#A", "GROUP#A#4", "GROUP#B#3", "GROUP#A#2", "GROUP#B#1"],
                "display": ["SPHERE", "SPHERE", "FIBER", "SPHERE", "FIBER"],
            }
        )
        colors = {"A": "#ff0000", "B": "#0000ff"}

        expected_data = [
            ("GROUP#A", "GROUP", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#A#2", "2", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#A#4", "4", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#B#1", "1", colors["B"], DISPLAY_TYPE.FIBER, ""),
            ("GROUP#B#3", "3", colors["B"], DISPLAY_TYPE.FIBER, ""),
        ]

        display_data = get_display_data(data, colors, url="", jitter=0.0)

        for expected, (key, display) in zip(expected_data, display_data.items()):
            self.assertTupleEqual(
                expected, (key, display.name, display.color, display.display_type, display.url)
            )

    def test_get_display_data_with_url(self):
        url = "https://url/"
        data = pd.DataFrame(
            {
                "name": ["GROUP#A#3#1", "GROUP#A#2#1", "GROUP#B#1#1", "GROUP#A#2#0", "GROUP#B#1#0"],
                "display": ["OBJ", "OBJ", "SPHERE", "OBJ", "OBJ"],
            }
        )
        colors = {"A": "#ff0000", "B": "#0000ff"}

        expected_data = [
            ("GROUP#A#2#0", "2", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000000_GROUP_002.MESH.obj"),
            ("GROUP#A#2#1", "2", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000001_GROUP_002.MESH.obj"),
            ("GROUP#A#3#1", "3", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000001_GROUP_003.MESH.obj"),
            ("GROUP#B#1#0", "1", colors["B"], DISPLAY_TYPE.OBJ, f"{url}/000000_GROUP_001.MESH.obj"),
            ("GROUP#B#1#1", "1", colors["B"], DISPLAY_TYPE.SPHERE, ""),
        ]

        display_data = get_display_data(data, colors, url=url, jitter=0.0)

        for expected, (key, display) in zip(expected_data, display_data.items()):
            self.assertTupleEqual(
                expected, (key, display.name, display.color, display.display_type, display.url)
            )

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_simularium"],
        "random",
        return_value=mock.Mock(),
    )
    def test_get_display_data_with_jitter(self, random_mock):
        random_mock.random.side_effect = [0.1, 0.3, 0.7, 0.9]

        data = pd.DataFrame(
            {
                "name": ["GROUP#A#1", "GROUP#A#2", "GROUP#A#3", "GROUP#A#4"],
                "display": ["SPHERE", "SPHERE", "SPHERE", "SPHERE"],
            }
        )
        jitter = 0.5
        color = "#ff55ee"
        colors = {"A": color}

        expected_colors = [
            shade_color(color, -0.2 * jitter),
            shade_color(color, -0.1 * jitter),
            shade_color(color, 0.1 * jitter),
            shade_color(color, 0.2 * jitter),
        ]

        display_data = get_display_data(data, colors, url="", jitter=jitter)

        for expected_color, display in zip(expected_colors, display_data.values()):
            self.assertEqual(expected_color, display.color)

    def test_shade_color(self):
        original_color = "#F0F00F"
        parameters = [
            (0.0, "#F0F00F"),  # unchanged
            (-1.0, "#000000"),  # full shade to black
            (1.0, "#FFFFFF"),  # full shade to white
            (-0.5, "#787808"),  # half shade to black
            (0.5, "#F8F887"),  # half shade to white
        ]

        for alpha, expected_color in parameters:
            with self.subTest(alpha=alpha):
                color = shade_color(original_color, alpha)
                self.assertEqual(expected_color.lower(), color.lower())


if __name__ == "__main__":
    unittest.main()
