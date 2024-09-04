import json
import tarfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from arcade_collection.output.parse_growth_file import (
    CELL_STATES,
    parse_growth_file,
    parse_growth_timepoint,
)


class TestParseGrowthFile(unittest.TestCase):
    def test_parse_growth_file(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        first_member_mock = mock.Mock(spec=tarfile.ExFileObject)
        second_member_mock = mock.Mock(spec=tarfile.ExFileObject)

        first_member_mock.name = "first_member.json"
        second_member_mock.name = "second_member.json"

        contents = {
            first_member_mock.name: first_member_mock,
            second_member_mock.name: second_member_mock,
        }

        tar_mock.getmembers.return_value = [*list(contents.values()), None]
        tar_mock.extractfile.side_effect = lambda member: (
            None if member is None else contents[member.name]
        )

        first_member_contents = {
            "seed": 0,
            "timepoints": [
                {
                    "time": 0.0,
                    "cells": [
                        [[-33, 0, 33, 0], [[0, 1, 2, 0, 2322.26, []]]],
                        [[0, 0, 10, 0], [[1, 0, 2, 0, 2300.50, []]]],
                    ],
                },
                {
                    "time": 0.5,
                    "cells": [
                        [[-33, 0, 31, 0], [[0, 1, 2, 0, 2522.26, []]]],
                        [[0, 0, 5, 0], [[1, 0, 3, 0, 4391.91, []]]],
                    ],
                },
                {
                    "time": 1.0,
                    "cells": [
                        [[-19, 0, 30, 0], [[0, 1, 1, 0, 2582.22, []]]],
                        [[0, 0, 7, 0], [[1, 0, 4, 0, 5047.58, [800.0, 512.3]]]],
                        [
                            [3, 3, -6, 0],
                            [[0, 1, 2, 0, 2453.83, [640.0]], [1, 0, 3, 1, 2517.54, []]],
                        ],
                    ],
                },
            ],
        }
        second_member_contents = {
            "seed": 1,
            "timepoints": [
                {
                    "time": 10.0,
                    "cells": [
                        [[-13, 0, 33, 0], [[0, 1, 2, 0, 2372.26, []]]],
                        [[0, 0, 10, 0], [[1, 0, 2, 0, 2390.50, []]]],
                    ],
                },
                {
                    "time": 10.5,
                    "cells": [
                        [[-33, 0, 1, 0], [[0, 1, 2, 0, 2022.26, []]]],
                        [[0, 0, 8, 0], [[1, 0, 3, 0, 4390.91, []]]],
                    ],
                },
                {
                    "time": 11.0,
                    "cells": [
                        [[-19, 0, 3, 0], [[0, 1, 1, 0, 2582.22, []]]],
                        [[1, 0, 1, 0], [[1, 0, 4, 0, 5040.58, [800.0, 512.3]]]],
                        [
                            [3, 0, -6, 0],
                            [[0, 2, 2, 0, 2053.83, [640.0]], [1, 0, 6, 1, 2517.54, []]],
                        ],
                    ],
                },
            ],
        }

        first_member_mock.read.return_value = json.dumps(first_member_contents).encode("utf-8")
        second_member_mock.read.return_value = json.dumps(second_member_contents).encode("utf-8")

        expected_data = {
            "TICK": [
                0.0,
                0.0,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
                10.0,
                10.5,
                10.5,
                11.0,
                11.0,
                11.0,
                11.0,
            ],
            "SEED": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "U": [-33, 0, -33, 0, -19, 0, 3, 3, -13, 0, -33, 0, -19, 1, 3, 3],
            "V": [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            "W": [33, 10, 31, 5, 30, 7, -6, -6, 33, 10, 1, 8, 3, 1, -6, -6],
            "Z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "POSITION": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "POPULATION": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0],
            "STATE": [
                "QUIESCENT",
                "QUIESCENT",
                "QUIESCENT",
                "MIGRATORY",
                "APOPTOTIC",
                "PROLIFERATIVE",
                "QUIESCENT",
                "MIGRATORY",
                "QUIESCENT",
                "QUIESCENT",
                "QUIESCENT",
                "MIGRATORY",
                "APOPTOTIC",
                "PROLIFERATIVE",
                "QUIESCENT",
                "NECROTIC",
            ],
            "VOLUME": [
                2322.26,
                2300.5,
                2522.26,
                4391.91,
                2582.22,
                5047.58,
                2453.83,
                2517.54,
                2372.26,
                2390.50,
                2022.26,
                4390.91,
                2582.22,
                5040.58,
                2053.83,
                2517.54,
            ],
            "CYCLE": [
                None,
                None,
                None,
                None,
                None,
                np.mean([800.0, 512.3]),
                640.0,
                None,
                None,
                None,
                None,
                None,
                None,
                np.mean([800.0, 512.3]),
                640.0,
                None,
            ],
        }

        data = parse_growth_file(tar_mock)

        self.assertTrue(pd.DataFrame(expected_data).equals(data))

    def test_parse_growth_timepoint(self):
        time = 15.0
        seed = 3
        data = {
            "time": time,
            "cells": [
                [
                    [10, 20, 30, 40],
                    [
                        [0, 4, 1, 7, 100, []],
                        [0, 5, 2, 8, 200, [40, 50, 60, 70]],
                    ],
                ],
                [
                    [50, 60, 70, 80],
                    [
                        [0, 6, 3, 9, 300, []],
                    ],
                ],
            ],
        }

        expected = [
            [time, seed, 10, 20, 30, 40, 7, 4, CELL_STATES[1], 100, None],
            [time, seed, 10, 20, 30, 40, 8, 5, CELL_STATES[2], 200, 55],
            [time, seed, 50, 60, 70, 80, 9, 6, CELL_STATES[3], 300, None],
        ]

        parsed = parse_growth_timepoint(data, seed)

        self.assertCountEqual(expected, parsed)


if __name__ == "__main__":
    unittest.main()
