import tarfile

import unittest
from unittest import mock

from arcade_collection.output.parse_growth_file import parse_growth_file


class TestParseGrowthFile(unittest.TestCase):
    def test_parse_growth_timepoint(self):
        tar_object = mock.Mock(spec=tarfile.TarFile)
        tar_object.name = "tar_object_name.tar"
        assert tar_object.name == "tar_object_name.tar"

        first_tar_member = mock.Mock(spec=tarfile.TarInfo)
        first_tar_member.name = "first_member.json"
        assert first_tar_member.name == "first_member.json"

        second_tar_member = mock.Mock(spec=tarfile.TarInfo)
        second_tar_member.name = "second_member.txt"
        assert second_tar_member.name == "second_member.txt"

        third_tar_member = mock.Mock(spec=tarfile.TarInfo)
        third_tar_member.name = "third_member.json"
        assert third_tar_member.name == "third_member.json"

        first_json = mock.MagicMock()
        first_json.read.return_value = '{"seed": 0, "timepoints": [{"time": 0.0,"cells": [[[-33,0,33,0],[[0,1,2,0,2322.26,[]]]],[[0,0,10,0],[[1,0,2,0,2300.50,[]]]]]},{"time": 0.5,"cells": [[[-33,0,31,0],[[0,1,2,0,2522.26,[]]]],[[0,0,5,0],[[1,0,3,0,4391.91,[]]]]]},{"time": 1.0,"cells": [[[-19,0,30,0],[[0,1,1,0,2582.22,[]]]],[[0,0,7,0],[[1,0,4,0,5047.58,[800.0,512.3]]]],[[3,3,-6,0],[[0,1,2,0,2453.83,[640.0]],[1,0,3,1,2517.54,[]]]]]}]}'.encode(
            "utf-8"
        )

        second_json = mock.MagicMock()
        second_json.read.return_value = '{"seed": 1, "timepoints": [{"time": 10.0,"cells": [[[-13,0,33,0],[[0,1,2,0,2372.26,[]]]],[[0,0,10,0],[[1,0,2,0,2390.50,[]]]]]},{"time": 10.5,"cells": [[[-33,0,1,0],[[0,1,2,0,2022.26,[]]]],[[0,0,8,0],[[1,0,3,0,4390.91,[]]]]]},{"time": 11.0,"cells": [[[-19,0,3,0],[[0,1,1,0,2582.22,[]]]],[[1,0,1,0],[[1,0,4,0,5040.58,[800.0,512.3]]]],[[3,0,-6,0],[[0,1,2,0,2053.83,[640.0]],[1,0,3,1,2517.54,[]]]]]}]}'.encode(
            "utf-8"
        )

        tar_object.getmembers.return_value = [first_tar_member, second_tar_member, third_tar_member]

        mock_contents = {
            first_tar_member: first_json,
            second_tar_member: "",
            third_tar_member: second_json,
        }
        tar_object.extractfile.side_effect = lambda fname, *args, **kwargs: mock_contents[fname]

        expected_dataframe = parse_growth_file.fn(tar_object)
        print(expected_dataframe)
