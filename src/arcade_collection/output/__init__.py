from prefect import task

from .convert_model_units import convert_model_units
from .convert_to_colorizer import convert_to_colorizer
from .convert_to_images import convert_to_images
from .convert_to_meshes import convert_to_meshes
from .convert_to_simularium import convert_to_simularium
from .extract_tick_json import extract_tick_json
from .get_location_voxels import get_location_voxels
from .merge_parsed_results import merge_parsed_results
from .parse_cells_file import parse_cells_file
from .parse_locations_file import parse_locations_file

convert_model_units = task(convert_model_units)
convert_to_colorizer = task(convert_to_colorizer)
convert_to_images = task(convert_to_images)
convert_to_meshes = task(convert_to_meshes)
convert_to_simularium = task(convert_to_simularium)
extract_tick_json = task(extract_tick_json)
get_location_voxels = task(get_location_voxels)
merge_parsed_results = task(merge_parsed_results)
parse_cells_file = task(parse_cells_file)
parse_locations_file = task(parse_locations_file)
