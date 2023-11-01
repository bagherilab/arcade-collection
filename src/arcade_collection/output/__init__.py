"""Tasks for processing output files from ARCADE."""

from prefect import task

from .convert_model_units import convert_model_units
from .extract_tick_json import extract_tick_json
from .get_location_voxels import get_location_voxels
from .merge_parsed_results import merge_parsed_results
from .parse_cells_file import parse_cells_file
from .parse_growth_file import parse_growth_file
from .parse_locations_file import parse_locations_file

convert_model_units = task(convert_model_units)
extract_tick_json = task(extract_tick_json)
get_location_voxels = task(get_location_voxels)
merge_parsed_results = task(merge_parsed_results)
parse_cells_file = task(parse_cells_file)
parse_growth_file = task(parse_growth_file)
parse_locations_file = task(parse_locations_file)
