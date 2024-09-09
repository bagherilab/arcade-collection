import sys
import unittest
from unittest import mock

import pandas as pd
from simulariumio import DISPLAY_TYPE

from arcade_collection.convert.convert_to_simularium import get_display_data, shade_color


class TestConvertToSimularium(unittest.TestCase):
    def test_get_display_data_no_url(self) -> None:
        data = pd.DataFrame(
            {
                "name": ["GROUP#A", "GROUP#A#4", "GROUP#B#3", "GROUP#A#2", "GROUP#B#1"],
                "display_type": ["SPHERE", "SPHERE", "FIBER", "SPHERE", "FIBER"],
            }
        )
        colors = {"A": "#ff0000", "B": "#0000ff"}

        expected_data = [
            ("GROUP#A", "GROUP", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#A#2", "2", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#A#4", "4", colors["A"], DISPLAY_TYPE.SPHERE, ""),
            ("GROUP#B#1", "1", colors["B"], DISPLAY_TYPE.FIBER, ""),
            ("GROUP#B#3", "3", colors["B"], DISPLAY_TYPE.FIBER, ""),
        ]

        display_data = get_display_data(data, colors, url="", jitter=0.0)

        for expected, (key, display) in zip(expected_data, display_data.items()):
            self.assertTupleEqual(
                expected, (key, display.name, display.color, display.display_type, display.url)
            )

    def test_get_display_data_with_url(self) -> None:
        url = "https://url/"
        data = pd.DataFrame(
            {
                "name": ["GROUP#A#3#1", "GROUP#A#2#1", "GROUP#B#1#1", "GROUP#A#2#0", "GROUP#B#1#0"],
                "display_type": ["OBJ", "OBJ", "OBJ", "OBJ", "OBJ"],
            }
        )
        colors = {"A": "#ff0000", "B": "#0000ff"}

        expected_data = [
            ("GROUP#A#2#0", "2", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000000_GROUP_002.MESH.obj"),
            ("GROUP#A#2#1", "2", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000001_GROUP_002.MESH.obj"),
            ("GROUP#A#3#1", "3", colors["A"], DISPLAY_TYPE.OBJ, f"{url}/000001_GROUP_003.MESH.obj"),
            ("GROUP#B#1#0", "1", colors["B"], DISPLAY_TYPE.OBJ, f"{url}/000000_GROUP_001.MESH.obj"),
            ("GROUP#B#1#1", "1", colors["B"], DISPLAY_TYPE.OBJ, f"{url}/000001_GROUP_001.MESH.obj"),
        ]

        display_data = get_display_data(data, colors, url=url, jitter=0.0)

        for expected, (key, display) in zip(expected_data, display_data.items()):
            self.assertTupleEqual(
                expected, (key, display.name, display.color, display.display_type, display.url)
            )

    @mock.patch.object(
        sys.modules["arcade_collection.convert.convert_to_simularium"],
        "random",
        return_value=mock.Mock(),
    )
    def test_get_display_data_with_jitter(self, random_mock) -> None:
        random_mock.random.side_effect = [0.1, 0.3, 0.7, 0.9]

        data = pd.DataFrame(
            {
                "name": ["GROUP#A#1", "GROUP#A#2", "GROUP#A#3", "GROUP#A#4"],
                "display_type": ["SPHERE", "SPHERE", "SPHERE", "SPHERE"],
            }
        )
        jitter = 0.5
        color = "#ff55ee"
        colors = {"A": color}

        expected_colors = [
            shade_color(color, -0.2 * jitter),
            shade_color(color, -0.1 * jitter),
            shade_color(color, 0.1 * jitter),
            shade_color(color, 0.2 * jitter),
        ]

        display_data = get_display_data(data, colors, url="", jitter=jitter)

        for expected_color, display in zip(expected_colors, display_data.values()):
            self.assertEqual(expected_color, display.color)

    def test_shade_color(self) -> None:
        original_color = "#F0F00F"
        parameters = [
            (0.0, "#F0F00F"),  # unchanged
            (-1.0, "#000000"),  # full shade to black
            (1.0, "#FFFFFF"),  # full shade to white
            (-0.5, "#787808"),  # half shade to black
            (0.5, "#F8F887"),  # half shade to white
        ]

        for alpha, expected_color in parameters:
            with self.subTest(alpha=alpha):
                color = shade_color(original_color, alpha)
                self.assertEqual(expected_color.lower(), color.lower())


if __name__ == "__main__":
    unittest.main()
