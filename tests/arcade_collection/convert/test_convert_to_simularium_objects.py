import json
import unittest

import numpy as np
import pandas as pd
from simulariumio.constants import VIZ_TYPE

from arcade_collection.convert.convert_to_simularium import CAMERA_LOOK_AT, CAMERA_POSITIONS
from arcade_collection.convert.convert_to_simularium_objects import convert_to_simularium_objects


class TestConvertToSimulariumObjects(unittest.TestCase):
    def test_convert_to_simularium_objects_potts(self) -> None:
        series_key = "SERIES_KEY"
        simulation_type = "potts"
        categories = pd.DataFrame(
            {
                "ID": [1, 3, 4, 1, 3, 4, 1, 4],
                "FRAME": [0, 0, 0, 5, 5, 5, 10, 10],
                "CATEGORY": ["A", "A", "A", "A", "B", "B", "B", "B"],
            }
        )
        frame_spec = (0, 11, 5)
        regions = ["REGION_A", "REGION_B"]
        box = (10, 12, 14)
        ds = (2, 3, 4)
        dt = 10
        colors = {"A": "#ff0000", "B": "#0000ff"}
        group_size = 2
        url = "URL"
        jitter = 0

        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        size = {"x": box[0] * ds[0], "y": box[1] * ds[1], "z": box[2] * ds[2]}
        camera_position = {
            "x": CAMERA_POSITIONS[simulation_type][0],
            "y": CAMERA_POSITIONS[simulation_type][1],
            "z": CAMERA_POSITIONS[simulation_type][2],
        }
        camera_look_at = {
            "x": CAMERA_LOOK_AT[simulation_type][0],
            "y": CAMERA_LOOK_AT[simulation_type][1],
            "z": CAMERA_LOOK_AT[simulation_type][2],
        }
        type_mapping_entries = [
            ("REGION_A", "A", 0, 0),
            ("REGION_B", "A", 0, 0),
            ("REGION_A", "A", 1, 0),
            ("REGION_B", "A", 1, 0),
            ("REGION_A", "A", 0, 5),
            ("REGION_B", "A", 0, 5),
            ("REGION_A", "B", 1, 5),
            ("REGION_B", "B", 1, 5),
            ("REGION_A", "B", 0, 10),
            ("REGION_B", "B", 0, 10),
        ]
        type_mapping = {
            str(i): {
                "name": f"{region}#{category}#{index}#{frame}",
                "geometry": {
                    "displayType": "OBJ",
                    "url": f"{url}/{frame:06d}_{region}_{index:03d}.MESH.obj",
                    "color": colors[category],
                },
            }
            for i, (region, category, index, frame) in enumerate(type_mapping_entries)
        }
        title = f"ARCADE - {series_key}"
        model_info = {
            "title": "ARCADE",
            "version": simulation_type,
            "description": f"Agent-based modeling framework ARCADE for {series_key}.",
        }

        def make_bundle_data(object_id, display_index):
            position = (0, 0, 0)
            rotation = (0, 0, 0)
            radius = 1
            subpoints = 0

            return [
                VIZ_TYPE.DEFAULT,
                object_id,
                display_index,
                *position,
                *rotation,
                radius,
                subpoints,
            ]

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0),
                    *make_bundle_data(1, 1),
                    *make_bundle_data(2, 2),
                    *make_bundle_data(3, 3),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 4),
                    *make_bundle_data(1, 5),
                    *make_bundle_data(2, 6),
                    *make_bundle_data(3, 7),
                ],
            },
            {
                "frameNumber": 2,
                "time": 2.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 8),
                    *make_bundle_data(1, 9),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_objects(
                series_key,
                simulation_type,
                categories,
                regions,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                group_size,
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

    def test_convert_to_simularium_objects_invalid_type_throws_exception(self) -> None:
        with self.assertRaises(ValueError):
            simulation_type = "invalid_type"
            convert_to_simularium_objects(
                "", simulation_type, None, [], (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, {}, 0, "", 0
            )


if __name__ == "__main__":
    unittest.main()
