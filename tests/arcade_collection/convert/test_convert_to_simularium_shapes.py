import json
import sys
import unittest
from math import sqrt
from unittest import mock

import numpy as np
from simulariumio.constants import VIZ_TYPE

from arcade_collection.convert.convert_to_simularium_shapes import (
    CELL_STATES,
    EDGE_TYPES,
    approximate_radius_from_voxels,
    calculate_patch_size,
    convert_hexagonal_to_rectangular_coordinates,
    convert_to_simularium_shapes,
)

from .utilities import build_tar_instance


class TestConvertToSimulariumShapes(unittest.TestCase):
    def test_convert_to_simularium_shapes_patch_invalid_data(self) -> None:
        simulation_type = "patch"
        simularium = convert_to_simularium_shapes(
            "", simulation_type, {"invalid": None}, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, {}
        )
        self.assertEqual("", simularium)

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_simularium_shapes"],
        "random",
        return_value=mock.Mock(),
    )
    def test_convert_to_simularium_shapes_patch_with_cells(self, random_mock) -> None:
        series_key = "SERIES_KEY"

        contents = {
            f"{series_key}.json": {
                "timepoints": [
                    {
                        "time": 0.0,
                        "cells": [
                            [[0, 1, -1, 0], [[0, 1, 0, 4, 27, []]]],
                            [[0, 1, -1, 1], [[0, 0, 1, 3, 216, []]]],
                        ],
                    },
                    {
                        "time": 5.0,
                        "cells": [
                            [[1, 0, -1, 2], [[0, 1, 2, 2, 729, []]]],
                            [[-1, 0, 1, 3], [[0, 0, 3, 1, 1728, []]]],
                        ],
                    },
                    {
                        "time": 10.0,
                        "cells": [
                            [[0, 0, 0, 4], [[0, 1, 4, 0, 3375, []]]],
                        ],
                    },
                ]
            }
        }

        simulation_type = "patch"
        data_tars = {"cells": build_tar_instance(contents)}
        frame_spec = (0, 11, 5)
        box = (2, 0, 0)
        ds = (2, 3, 4)
        dt = 10
        colors = {
            CELL_STATES[0]: "#ff0000",
            CELL_STATES[1]: "#00ff00",
            CELL_STATES[2]: "#0000ff",
            CELL_STATES[3]: "#ff00ff",
            CELL_STATES[4]: "#00ffff",
        }
        resolution = 0
        jitter = 0

        random_mock.randint.side_effect = [0, 1, 2, 3, 4]

        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        bounds, length, width = calculate_patch_size(box[0], box[1])
        size = {"x": length * ds[0], "y": width * ds[1], "z": box[2] * ds[2]}
        type_mapping = {
            str(i): {
                "name": f"{population}#{state}#{u}{v}{w}{z}{p}",
                "geometry": {"displayType": "SPHERE", "color": colors[state]},
            }
            for i, (population, u, v, w, z, p, state) in enumerate(
                [
                    ("POPULATION1", 0, 1, -1, 0, 4, CELL_STATES[0]),
                    ("POPULATION0", 0, 1, -1, 1, 3, CELL_STATES[1]),
                    ("POPULATION1", 1, 0, -1, 2, 2, CELL_STATES[2]),
                    ("POPULATION0", -1, 0, 1, 3, 1, CELL_STATES[3]),
                    ("POPULATION1", 0, 0, 0, 4, 0, CELL_STATES[4]),
                ]
            )
        }

        def make_bundle_data(object_id, display_index, radius, uvw, z, offset):
            rotation = (0, 0, 0)
            subpoints = 0
            x, y = convert_hexagonal_to_rectangular_coordinates(uvw, bounds, offset)

            return [
                VIZ_TYPE.DEFAULT,
                object_id,
                display_index,
                (x - length / 2.0) * ds[0],
                (width / 2.0 - y) * ds[1],
                z * ds[2],
                *rotation,
                radius,
                subpoints,
            ]

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0, 2, (0, 1, -1), 0, 4),
                    *make_bundle_data(1, 1, 4, (0, 1, -1), 1, 4),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 2, 6, (1, 0, -1), 2, 4),
                    *make_bundle_data(1, 3, 8, (-1, 0, 1), 3, 4),
                ],
            },
            {
                "frameNumber": 2,
                "time": 2.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 4, 10, (0, 0, 0), 4, 4),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_shapes(
                series_key,
                simulation_type,
                data_tars,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                resolution,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_simularium_shapes"],
        "random",
        return_value=mock.Mock(),
    )
    def test_convert_to_simularium_shapes_patch_with_graph(self, random_mock) -> None:
        series_key = "SERIES_KEY"

        contents = {
            f"{series_key}.GRAPH.json": {
                "timepoints": [
                    {
                        "time": 0.0,
                        "graph": [
                            [
                                [1 * sqrt(3), 2, 3, 0, 0],
                                [4 * sqrt(3), 5, 6, 0, 0],
                                [-2, 7, 0, 0, 0, 0, 1],
                            ],
                            [
                                [8 * sqrt(3), 9, 10, 0, 0],
                                [11 * sqrt(3), 12, 13, 0, 0],
                                [-1, 14, 0, 0, 0, 0, 1],
                            ],
                        ],
                    },
                    {
                        "time": 5.0,
                        "graph": [
                            [
                                [15 * sqrt(3), 16, 17, 0, 0],
                                [18 * sqrt(3), 19, 20, 0, 0],
                                [0, 21, 0, 0, 0, 0, 1],
                            ],
                            [
                                [22 * sqrt(3), 23, 24, 0, 0],
                                [25 * sqrt(3), 26, 27, 0, 0],
                                [1, 28, 0, 0, 0, 0, 1],
                            ],
                        ],
                    },
                    {
                        "time": 10.0,
                        "graph": [
                            [
                                [29 * sqrt(3), 30, 31, 0, 0],
                                [32 * sqrt(3), 33, 34, 0, 0],
                                [2, 35, 0, 0, 0, 0, 1],
                            ],
                            [
                                [36 * sqrt(3), 37, 38, 0, 0],
                                [39 * sqrt(3), 40, 41, 0, 0],
                                [2, 42, 0, 0, 0, 0, np.nan],
                            ],
                        ],
                    },
                ]
            }
        }

        simulation_type = "patch"
        data_tars = {"graph": build_tar_instance(contents)}
        frame_spec = (0, 11, 5)
        box = (2, 0, 0)
        ds = (2, 3, 4)
        dt = 10
        colors = {
            EDGE_TYPES[0]: "#ff0000",
            EDGE_TYPES[1]: "#00ff00",
            EDGE_TYPES[2]: "#0000ff",
            EDGE_TYPES[3]: "#ff00ff",
            EDGE_TYPES[4]: "#00ffff",
            EDGE_TYPES[5]: "#ffff00",
        }
        resolution = 0
        jitter = 0

        random_mock.randint.side_effect = [0, 1, 2, 3, 4]

        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        _, length, width = calculate_patch_size(box[0], box[1])
        size = {"x": length * ds[0], "y": width * ds[1], "z": box[2] * ds[2]}
        type_mapping = {
            str(i): {
                "name": f"VASCULATURE#{edge_type}",
                "geometry": {"displayType": "FIBER", "color": colors[edge_type]},
            }
            for i, edge_type in enumerate(EDGE_TYPES)
        }

        def make_bundle_data(object_id, display_index, radius, length, width, point1, point2):
            rotation = (0, 0, 0)
            subpoints = 6
            x1, y1, z1 = point1
            x2, y2, z2 = point2

            return [
                VIZ_TYPE.FIBER,
                object_id,
                display_index,
                -length / 2 * ds[0],
                width / 2 * ds[1],
                0,
                *rotation,
                radius,
                subpoints,
                x1 * ds[0],
                -y1 * ds[1],
                z1 * ds[2],
                x2 * ds[0],
                -y2 * ds[1],
                z2 * ds[2],
            ]

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0, 7, length, width, (1, 2, 3), (4, 5, 6)),
                    *make_bundle_data(1, 1, 14, length, width, (8, 9, 10), (11, 12, 13)),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 2, 21, length, width, (15, 16, 17), (18, 19, 20)),
                    *make_bundle_data(1, 3, 28, length, width, (22, 23, 24), (25, 26, 27)),
                ],
            },
            {
                "frameNumber": 2,
                "time": 2.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 4, 35, length, width, (29, 30, 31), (32, 33, 34)),
                    *make_bundle_data(1, 5, 42, length, width, (36, 37, 38), (39, 40, 41)),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_shapes(
                series_key,
                simulation_type,
                data_tars,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                resolution,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    def test_convert_to_simularium_shapes_potts_invalid_data(self) -> None:
        simulation_type = "potts"
        simularium = convert_to_simularium_shapes(
            "", simulation_type, {"invalid": None}, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, {}
        )
        self.assertEqual("", simularium)

    def test_convert_to_simularium_shapes_potts_resolution_zero(self) -> None:
        series_key = "SERIES_KEY"

        cells_contents = {
            f"{series_key}_000000.CELLS.json": [{"id": 1, "phase": "A"}],
            f"{series_key}_000005.CELLS.json": [{"id": 2, "phase": "B"}, {"id": 3, "phase": "A"}],
            f"{series_key}_000010.CELLS.json": [{"id": 4, "phase": "B"}],
        }

        locs_contents = {
            f"{series_key}_000000.LOCATIONS.json": [
                {
                    "id": 1,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[1, 2, 3], [2, 2, 3], [5, 4, 0]]},
                        {"region": "REGION_A", "voxels": [[1, 3, 3], [1, 4, 6]]},
                    ],
                }
            ],
            f"{series_key}_000005.LOCATIONS.json": [
                {
                    "id": 2,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[2, 4, 6], [4, 4, 6], [10, 8, 0]]},
                        {"region": "REGION_B", "voxels": [[2, 6, 6], [2, 8, 12]]},
                    ],
                },
                {
                    "id": 3,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[3, 6, 9], [6, 6, 9], [15, 12, 0]]},
                        {"region": "REGION_C", "voxels": [[3, 9, 9], [3, 12, 18]]},
                    ],
                },
            ],
            f"{series_key}_000010.LOCATIONS.json": [
                {
                    "id": 4,
                    "location": [
                        {"region": "DEFAULT", "voxels": [[4, 8, 12], [8, 8, 12], [20, 16, 0]]},
                        {"region": "REGION_D", "voxels": [[4, 12, 12], [4, 16, 24]]},
                    ],
                },
            ],
        }

        simulation_type = "potts"
        data_tars = {
            "cells": build_tar_instance(cells_contents),
            "locations": build_tar_instance(locs_contents),
        }
        frame_spec = (0, 11, 5)
        box = (10, 20, 30)
        ds = (2, 3, 4)
        dt = 10
        colors = {"A": "#ff0000", "B": "#00ff00"}
        resolution = 0
        jitter = 0

        length, width, height = box
        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        size = {"x": length * ds[0], "y": width * ds[1], "z": height * ds[2]}
        type_mapping = {
            str(i): {
                "name": f"{region}#{phase}#{index}",
                "geometry": {"displayType": "SPHERE", "color": colors[phase]},
            }
            for i, (region, phase, index) in enumerate(
                [
                    ("DEFAULT", "A", 1),
                    ("REGION_A", "A", 1),
                    ("DEFAULT", "B", 2),
                    ("REGION_B", "B", 2),
                    ("DEFAULT", "A", 3),
                    ("REGION_C", "A", 3),
                    ("DEFAULT", "B", 4),
                    ("REGION_D", "B", 4),
                ]
            )
        }

        def make_bundle_data(object_id, display_index, center, voxels):
            cx, cy, cz = center
            rotation = (0, 0, 0)
            subpoints = 0

            return [
                VIZ_TYPE.DEFAULT,
                object_id,
                display_index,
                (cx - length / 2.0) * ds[0],
                (width / 2.0 - cy) * ds[1],
                (cz - height / 2.0) * ds[2],
                *rotation,
                approximate_radius_from_voxels(voxels),
                subpoints,
            ]

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0, (2, 3, 3), 5),
                    *make_bundle_data(1, 1, (1, 3.5, 4.5), 2),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 2, (4, 6, 6), 5),
                    *make_bundle_data(1, 3, (2, 7, 9), 2),
                    *make_bundle_data(2, 4, (6, 9, 9), 5),
                    *make_bundle_data(3, 5, (3, 10.5, 13.5), 2),
                ],
            },
            {
                "frameNumber": 2,
                "time": 2.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 6, (8, 12, 12), 5),
                    *make_bundle_data(1, 7, (4, 14, 18), 2),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_shapes(
                series_key,
                simulation_type,
                data_tars,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                resolution,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    def test_convert_to_simularium_shapes_potts_resolution_one(self) -> None:
        series_key = "SERIES_KEY"

        hollow_cube_voxels = [[x, y, z] for x in range(3) for y in range(3) for z in range(3)]
        full_cube_voxels = [[0, 0, 0]]

        expected_hollow_cube_voxels = [
            [x, y, z]
            for x in range(3)
            for y in range(3)
            for z in range(3)
            if [x, y, z] != [1, 1, 1]
        ]
        expected_full_cube_voxels = [[0, 0, 0]]

        cells_contents = {
            f"{series_key}_000000.CELLS.json": [{"id": 1, "phase": "A"}],
            f"{series_key}_000005.CELLS.json": [{"id": 2, "phase": "B"}],
        }

        locs_contents = {
            f"{series_key}_000000.LOCATIONS.json": [
                {
                    "id": 1,
                    "location": [
                        {
                            "region": "UNDEFINED",
                            "voxels": hollow_cube_voxels,
                        }
                    ],
                }
            ],
            f"{series_key}_000005.LOCATIONS.json": [
                {
                    "id": 2,
                    "location": [
                        {
                            "region": "UNDEFINED",
                            "voxels": full_cube_voxels,
                        }
                    ],
                }
            ],
        }

        simulation_type = "potts"
        data_tars = {
            "cells": build_tar_instance(cells_contents),
            "locations": build_tar_instance(locs_contents),
        }
        frame_spec = (0, 6, 5)
        box = (10, 20, 30)
        ds = (2, 3, 4)
        dt = 10
        colors = {"A": "#ff0000", "B": "#00ff00"}
        resolution = 1
        jitter = 0

        length, width, height = box
        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        size = {"x": length * ds[0], "y": width * ds[1], "z": height * ds[2]}
        type_mapping = {
            "0": {
                "name": "UNDEFINED#A#1",
                "geometry": {"displayType": "SPHERE", "color": colors["A"]},
            },
            "1": {
                "name": "UNDEFINED#B#2",
                "geometry": {"displayType": "SPHERE", "color": colors["B"]},
            },
        }

        def make_bundle_data(object_id, display_index, voxels):
            bundle_data = []
            rotation = (0, 0, 0)
            subpoints = 0

            for i, (x, y, z) in enumerate(voxels):
                bundle_data.extend(
                    [
                        VIZ_TYPE.DEFAULT,
                        object_id + i,
                        display_index,
                        (x - length / 2.0) * ds[0],
                        (width / 2.0 - y) * ds[1],
                        (z - height / 2.0) * ds[2],
                        *rotation,
                        resolution / 2,
                        subpoints,
                    ]
                )

            return bundle_data

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0, expected_hollow_cube_voxels),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 1, expected_full_cube_voxels),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_shapes(
                series_key,
                simulation_type,
                data_tars,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                resolution,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    def test_convert_to_simularium_shapes_potts_resolution_two(self) -> None:
        series_key = "SERIES_KEY"

        hollow_cube_voxels = [[x, y, z] for x in range(6) for y in range(6) for z in range(6)]
        full_cube_voxels = [[x, y, z] for x in range(4) for y in range(4) for z in range(4)]

        expected_hollow_cube_voxels = [
            [2 * x + 0.5, 2 * y + 0.5, 2 * z + 0.5]
            for x in range(3)
            for y in range(3)
            for z in range(3)
            if [x, y, z] != [1, 1, 1]
        ]
        expected_full_cube_voxels = [
            [2 * x + 0.5, 2 * y + 0.5, 2 * z + 0.5]
            for x in range(2)
            for y in range(2)
            for z in range(2)
        ]

        cells_contents = {
            f"{series_key}_000000.CELLS.json": [{"id": 1, "phase": "A"}],
            f"{series_key}_000005.CELLS.json": [{"id": 2, "phase": "B"}],
        }

        locs_contents = {
            f"{series_key}_000000.LOCATIONS.json": [
                {
                    "id": 1,
                    "location": [
                        {
                            "region": "UNDEFINED",
                            "voxels": hollow_cube_voxels,
                        }
                    ],
                }
            ],
            f"{series_key}_000005.LOCATIONS.json": [
                {
                    "id": 2,
                    "location": [
                        {
                            "region": "UNDEFINED",
                            "voxels": full_cube_voxels,
                        }
                    ],
                }
            ],
        }

        simulation_type = "potts"
        data_tars = {
            "cells": build_tar_instance(cells_contents),
            "locations": build_tar_instance(locs_contents),
        }
        frame_spec = (0, 6, 5)
        box = (10, 20, 30)
        ds = (2, 3, 4)
        dt = 10
        colors = {"A": "#ff0000", "B": "#00ff00"}
        resolution = 2
        jitter = 0

        length, width, height = box
        start_time, end_time, time_interval = frame_spec
        total_steps = len(np.arange(start_time, end_time, time_interval))
        time_step_size = dt * time_interval
        size = {"x": length * ds[0], "y": width * ds[1], "z": height * ds[2]}
        type_mapping = {
            "0": {
                "name": "UNDEFINED#A#1",
                "geometry": {"displayType": "SPHERE", "color": colors["A"]},
            },
            "1": {
                "name": "UNDEFINED#B#2",
                "geometry": {"displayType": "SPHERE", "color": colors["B"]},
            },
        }

        def make_bundle_data(object_id, display_index, voxels):
            bundle_data = []
            rotation = (0, 0, 0)
            subpoints = 0

            for i, (x, y, z) in enumerate(voxels):
                bundle_data.extend(
                    [
                        VIZ_TYPE.DEFAULT,
                        object_id + i,
                        display_index,
                        (x - length / 2.0) * ds[0],
                        (width / 2.0 - y) * ds[1],
                        (z - height / 2.0) * ds[2],
                        *rotation,
                        resolution / 2,
                        subpoints,
                    ]
                )

            return bundle_data

        bundle_data = [
            {
                "frameNumber": 0,
                "time": 0.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 0, expected_hollow_cube_voxels),
                ],
            },
            {
                "frameNumber": 1,
                "time": 1.0 * time_step_size,
                "data": [
                    *make_bundle_data(0, 1, expected_full_cube_voxels),
                ],
            },
        ]

        simularium = json.loads(
            convert_to_simularium_shapes(
                series_key,
                simulation_type,
                data_tars,
                frame_spec,
                box,
                ds,
                dt,
                colors,
                resolution,
                jitter,
            )
        )

        trajectory_info = simularium["trajectoryInfo"]

        self.assertEqual(total_steps, trajectory_info["totalSteps"])
        self.assertEqual(time_step_size, trajectory_info["timeStepSize"])
        self.assertDictEqual(size, trajectory_info["size"])
        self.assertDictEqual(type_mapping, trajectory_info["typeMapping"])
        self.assertListEqual(bundle_data, simularium["spatialData"]["bundleData"])

    def test_convert_to_simularium_shapes_invalid_type_throws_exception(self) -> None:
        with self.assertRaises(ValueError):
            simulation_type = "invalid_type"
            convert_to_simularium_shapes(
                "", simulation_type, {}, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, {}
            )


if __name__ == "__main__":
    unittest.main()
