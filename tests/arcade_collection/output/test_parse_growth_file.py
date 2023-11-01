import tarfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from arcade_collection.output.parse_growth_file import parse_growth_file


class TestParseGrowthFile(unittest.TestCase):
    def test_parse_growth_timepoint(self):
        tar_object = mock.Mock(spec=tarfile.TarFile)
        tar_object.name = "tar_object_name.tar.xz"

        first_tar_member = mock.Mock(spec=tarfile.TarInfo)
        first_tar_member.name = "first_member.json"

        second_tar_member = mock.Mock(spec=tarfile.TarInfo)
        second_tar_member.name = "second_member.json"

        tar_object.getmembers.return_value = [first_tar_member, second_tar_member]

        first_json = mock.MagicMock()
        first_json.read.return_value = '{"seed": 0, "timepoints": [{"time": 0.0,"cells": [[[-33,0,33,0],[[0,1,2,0,2322.26,[]]]],[[0,0,10,0],[[1,0,2,0,2300.50,[]]]]]},{"time": 0.5,"cells": [[[-33,0,31,0],[[0,1,2,0,2522.26,[]]]],[[0,0,5,0],[[1,0,3,0,4391.91,[]]]]]},{"time": 1.0,"cells": [[[-19,0,30,0],[[0,1,1,0,2582.22,[]]]],[[0,0,7,0],[[1,0,4,0,5047.58,[800.0,512.3]]]],[[3,3,-6,0],[[0,1,2,0,2453.83,[640.0]],[1,0,3,1,2517.54,[]]]]]}]}'.encode(
            "utf-8"
        )

        second_json = mock.MagicMock()
        second_json.read.return_value = '{"seed": 1, "timepoints": [{"time": 10.0,"cells": [[[-13,0,33,0],[[0,1,2,0,2372.26,[]]]],[[0,0,10,0],[[1,0,2,0,2390.50,[]]]]]},{"time": 10.5,"cells": [[[-33,0,1,0],[[0,1,2,0,2022.26,[]]]],[[0,0,8,0],[[1,0,3,0,4390.91,[]]]]]},{"time": 11.0,"cells": [[[-19,0,3,0],[[0,1,1,0,2582.22,[]]]],[[1,0,1,0],[[1,0,4,0,5040.58,[800.0,512.3]]]],[[3,0,-6,0],[[0,2,2,0,2053.83,[640.0]],[1,0,6,1,2517.54,[]]]]]}]}'.encode(
            "utf-8"
        )

        mock_contents = {
            first_tar_member: first_json,
            second_tar_member: second_json,
        }
        tar_object.extractfile.side_effect = lambda fname, *args, **kwargs: mock_contents[fname]

        returned_df = parse_growth_file(tar_object)

        expected_dict = {
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

        expected_df = pd.DataFrame(expected_dict)
        self.assertTrue(expected_df.equals(returned_df))
