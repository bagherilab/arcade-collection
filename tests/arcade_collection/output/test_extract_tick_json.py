import json
import tarfile
import unittest
from unittest import mock

from arcade_collection.output.extract_tick_json import extract_tick_json


class TestExtractTickJson(unittest.TestCase):
    def test_extract_tick_json_file_does_not_exist(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        tar_mock.extractfile.return_value = None

        with self.assertRaises(AssertionError):
            extract_tick_json(tar_mock, "", 0)

    def test_extract_tick_json_integer_tick_without_extension(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        file_mock = mock.Mock(spec=tarfile.ExFileObject)

        contents = [{"id": 1, "state": "STATE_A"}, {"id": 2, "state": "STATE_B"}]

        key = "KEY"
        tick = 10
        file_name = f"{key}_{tick:06d}.json"

        tar_mock.extractfile.side_effect = lambda name: file_mock if name == file_name else None
        file_mock.read.return_value = json.dumps(contents).encode("utf-8")

        extracted = extract_tick_json(tar_mock, key, tick)

        self.assertListEqual(contents, extracted)

    def test_extract_tick_json_integer_tick_with_extension(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        file_mock = mock.Mock(spec=tarfile.ExFileObject)

        contents = [{"id": 1, "state": "STATE_A"}, {"id": 2, "state": "STATE_B"}]

        key = "KEY"
        tick = 10
        extension = "EXT"
        file_name = f"{key}_{tick:06d}.{extension}.json"

        tar_mock.extractfile.side_effect = lambda name: file_mock if name == file_name else None
        file_mock.read.return_value = json.dumps(contents).encode("utf-8")

        extracted = extract_tick_json(tar_mock, key, tick, extension=extension)

        self.assertListEqual(contents, extracted)

    def test_extract_tick_json_float_tick_without_extension(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        file_mock = mock.Mock(spec=tarfile.ExFileObject)

        contents = {
            "timepoints": [
                {"time": 1.0, "field": ["a", "b", "c"]},
                {"time": 2.0, "field": ["d", "e", "f"]},
                {"time": 3.0, "field": ["g", "h", "i"]},
            ]
        }

        key = "KEY"
        tick = 2.0
        file_name = f"{key}.json"

        tar_mock.extractfile.side_effect = lambda name: file_mock if name == file_name else None
        file_mock.read.return_value = json.dumps(contents).encode("utf-8")

        extracted = extract_tick_json(tar_mock, key, tick, field="field")

        self.assertCountEqual(["d", "e", "f"], extracted)

    def test_extract_tick_json_float_tick_with_extension(self):
        tar_mock = mock.Mock(spec=tarfile.TarFile)
        file_mock = mock.Mock(spec=tarfile.ExFileObject)

        contents = {
            "timepoints": [
                {"time": 1.0, "field": ["a", "b", "c"]},
                {"time": 2.0, "field": ["d", "e", "f"]},
                {"time": 3.0, "field": ["g", "h", "i"]},
            ]
        }

        key = "KEY"
        tick = 2.0
        extension = "EXT"
        file_name = f"{key}.{extension}.json"

        tar_mock.extractfile.side_effect = lambda name: file_mock if name == file_name else None
        file_mock.read.return_value = json.dumps(contents).encode("utf-8")

        extracted = extract_tick_json(tar_mock, key, tick, extension=extension, field="field")

        self.assertCountEqual(["d", "e", "f"], extracted)


if __name__ == "__main__":
    unittest.main()
