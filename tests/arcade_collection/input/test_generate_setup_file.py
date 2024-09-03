import unittest

import pandas as pd

from arcade_collection.input.generate_setup_file import (
    DEFAULT_POPULATION_ID,
    calculate_sample_bounds,
    generate_setup_file,
    make_setup_file,
)


class TestGenerateSetupFile(unittest.TestCase):
    def setUp(self):
        self.terms = ["term_a", "term_b", "term_c"]
        self.margins = [10, 20, 30]

        self.setup_template_no_region = (
            "<set>\n"
            '    <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1" ticks="1"'
            ' length="%d" width="%d" height="%d">\n'
            "        <potts>\n"
            '            <potts.term id="term_a" />\n'
            '            <potts.term id="term_b" />\n'
            '            <potts.term id="term_c" />\n'
            "        </potts>\n"
            "        <agents>\n"
            "            <populations>\n"
            f'                <population id="{DEFAULT_POPULATION_ID}" init="%d" />\n'
            "            </populations>\n"
            "        </agents>\n"
            "    </series>\n"
            "</set>"
        )

        self.setup_template_with_region = (
            "<set>\n"
            '    <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1" ticks="1"'
            ' length="%d" width="%d" height="%d">\n'
            "        <potts>\n"
            '            <potts.term id="term_a" />\n'
            '            <potts.term id="term_b" />\n'
            '            <potts.term id="term_c" />\n'
            "        </potts>\n"
            "        <agents>\n"
            "            <populations>\n"
            f'                <population id="{DEFAULT_POPULATION_ID}" init="%d">\n'
            f'                    <population.region id="%s" />\n'
            f'                    <population.region id="%s" />\n'
            "                </population>\n"
            "            </populations>\n"
            "        </agents>\n"
            "    </series>\n"
            "</set>"
        )

    def test_generate_setup_file_no_regions(self):
        samples = pd.DataFrame(
            {
                "id": [1, 2],
                "x": [0, 2],
                "y": [3, 7],
                "z": [6, 7],
            }
        )

        expected_setup = self.setup_template_no_region % (25, 47, 64, 2)

        setup = generate_setup_file(samples, self.margins, self.terms)

        self.assertEqual(expected_setup, setup)

    def test_generate_setup_file_invalid_regions(self):
        samples = pd.DataFrame(
            {
                "id": [1, 2],
                "x": [0, 2],
                "y": [3, 7],
                "z": [6, 7],
                "region": [None, None],
            }
        )

        expected_setup = self.setup_template_no_region % (25, 47, 64, 2)

        setup = generate_setup_file(samples, self.margins, self.terms)

        self.assertEqual(expected_setup, setup)

    def test_generate_setup_file_with_regions(self):
        samples = pd.DataFrame(
            {
                "id": [1, 2],
                "x": [0, 2],
                "y": [3, 7],
                "z": [6, 7],
                "region": ["A", "B"],
            }
        )

        expected_setup = self.setup_template_with_region % (25, 47, 64, 2, "A", "B")

        setup = generate_setup_file(samples, self.margins, self.terms)

        self.assertEqual(expected_setup, setup)

    def test_calculate_sample_bounds(self):
        samples = pd.DataFrame(
            {
                "x": [0, 2],
                "y": [3, 7],
                "z": [6, 7],
            }
        )
        margins = [10, 20, 30]

        expected_bounds = (25, 47, 64)

        bounds = calculate_sample_bounds(samples, margins)

        self.assertTupleEqual(expected_bounds, bounds)

    def test_make_setup_file_no_regions(self):
        init = 100
        bounds = (10, 20, 30)
        regions = None

        expected_setup = self.setup_template_no_region % (*bounds, init)

        setup = make_setup_file(init, bounds, self.terms, regions)

        self.assertEqual(expected_setup, setup)

    def test_make_setup_file_with_regions(self):
        init = 100
        bounds = (10, 20, 30)
        regions = ["REGION_A", "REGION_B"]

        expected_setup = self.setup_template_with_region % (*bounds, init, regions[0], regions[1])

        setup = make_setup_file(init, bounds, self.terms, regions)

        self.assertEqual(expected_setup, setup)


if __name__ == "__main__":
    unittest.main()
