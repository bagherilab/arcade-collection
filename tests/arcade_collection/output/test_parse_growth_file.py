import json
import tarfile

import unittest
from unittest import mock
from unittest.mock import mock_open

from arcade_collection.output.parse_growth_file import parse_growth_file


class TestParseGrowthFile(unittest.TestCase):
    # @mock.patch("builtins.open", new_callable=mock_open)
    # @mock.patch("arcade_collection.output.parse_growth_file.json")
    def test_parse_growth_timepoint(self):
        tar_object = mock.Mock(spec=tarfile.TarFile)
        tar_object.name = "tar_object_name.tar"
        assert tar_object.name == "tar_object_name.tar"

        first_tar_member = mock.Mock(spec=tarfile.TarInfo)
        first_tar_member.name = "first_member.json"
        assert first_tar_member.name == "first_member.json"

        first_json = mock.MagicMock()
        first_json.read.return_value = '{"seed": 0, "timepoints": [{"time": 0.0,"cells": [[[-33,0,33,0],[[0,1,2,0,2522.26,[]]]],[[0,0,0,0],[[1,0,4,0,2300.50,[]]]]]},{"time": 0.5,"cells": [[[-33,0,33,0],[[0,1,2,0,2522.26,[]]]],[[0,0,0,0],[[1,0,4,0,4391.91,[]]]]]},{"time": 1.0,"cells": [[[-33,0,33,0],[[0,1,1,0,2522.26,[]]]],[[0,0,0,0],[[1,0,4,0,5047.58,[800.0,512.3]]]],[[3,3,-6,0],[[0,1,2,0,2453.83,[640.0]],[1,0,3,1,2517.54,[]]]]]}]}'.encode(
            "utf-8"
        )

        second_tar_member = mock.Mock(spec=tarfile.TarInfo)
        second_tar_member.name = "second_member.txt"
        assert second_tar_member.name == "second_member.txt"

        tar_object.getmembers.return_value = [first_tar_member, second_tar_member]
        tar_object.extractfile.return_value = first_json
        expected_dataframe = parse_growth_file.fn(tar_object)
        print(expected_dataframe)
