import unittest

from arcade_collection.convert.convert_to_simularium_objects import convert_to_simularium_objects


class TestConvertToSimulariumObjects(unittest.TestCase):
    def test_convert_to_simularium_objects_invalid_type_throws_exception(self) -> None:
        with self.assertRaises(ValueError):
            simulation_type = "invalid_type"
            convert_to_simularium_objects(
                "", simulation_type, None, (0, 0, 0), [], (0, 0, 0), 0, 0, 0, {}, 0, ""
            )


if __name__ == "__main__":
    unittest.main()
