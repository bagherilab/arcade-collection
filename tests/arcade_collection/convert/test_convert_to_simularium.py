import unittest

from arcade_collection.convert.convert_to_simularium import convert_to_simularium


class TestConvertToSimularium(unittest.TestCase):
    def test_convert_to_simularium_invalid_type_throws_exception(self) -> None:
        with self.assertRaises(ValueError):
            simulation_type = "invalid_type"
            convert_to_simularium("", simulation_type, {}, (0, 0, 0), (0, 0, 0), 0, 0, 0, {})


if __name__ == "__main__":
    unittest.main()
