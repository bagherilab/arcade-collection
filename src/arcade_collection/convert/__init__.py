"""Tasks for converting ARCADE files to other formats."""

from prefect import task

from .convert_to_colorizer import convert_to_colorizer
from .convert_to_images import convert_to_images
from .convert_to_meshes import convert_to_meshes
from .convert_to_projection import convert_to_projection
from .convert_to_simularium import convert_to_simularium

convert_to_colorizer = task(convert_to_colorizer)
convert_to_images = task(convert_to_images)
convert_to_meshes = task(convert_to_meshes)
convert_to_projection = task(convert_to_projection)
convert_to_simularium = task(convert_to_simularium)
