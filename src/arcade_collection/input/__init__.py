import importlib
import sys

from prefect import task

from .convert_to_cells_file import convert_to_cells_file
from .convert_to_locations_file import convert_to_locations_file
from .generate_setup_file import generate_setup_file
from .group_template_conditions import group_template_conditions
from .merge_region_samples import merge_region_samples

TASK_MODULES = [
    convert_to_cells_file,
    convert_to_locations_file,
    generate_setup_file,
    group_template_conditions,
    merge_region_samples,
]

for task_module in TASK_MODULES:
    MODULE_NAME = task_module.__name__
    module = importlib.import_module(f".{MODULE_NAME}", package=__name__)
    setattr(sys.modules[__name__], MODULE_NAME, task(getattr(module, MODULE_NAME)))
