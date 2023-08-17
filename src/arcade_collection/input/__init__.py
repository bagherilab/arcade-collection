"""Tasks for generating input files for ARCADE."""

from prefect import task

from .convert_to_cells_file import convert_to_cells_file
from .convert_to_locations_file import convert_to_locations_file
from .generate_setup_file import generate_setup_file
from .group_template_conditions import group_template_conditions
from .merge_region_samples import merge_region_samples

convert_to_cells_file = task(convert_to_cells_file)
convert_to_locations_file = task(convert_to_locations_file)
generate_setup_file = task(generate_setup_file)
group_template_conditions = task(group_template_conditions)
merge_region_samples = task(merge_region_samples)
