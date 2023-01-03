import unittest
from unittest import mock

from arcade_collection.output.parse_growth_file import parse_growth_file


class TestParseGrowthFile(unittest.TestCase):
    @mock.patch("arcade_collection.output.parse_growth_file.tarfile")
    def test_parse_growth_timepoint(self, tar_mock):
        tar_object = mock.Mock(spec=tar_mock.TarFile)
