# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib.metadata
import os
import sys
from datetime import datetime

from sphinx.ext.autosummary.generate import AutosummaryRenderer

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "ARCADE Collection"
author = importlib.metadata.metadata("arcade-collection")["Author"]
copyright = f"{datetime.now().year}, {author}, Bagheri Lab"
release = importlib.metadata.version("arcade-collection")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Decides whether module names are prepended to all object names.
add_module_names = False

# Default options for autodoc directives.
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
}

# List of modules to be mocked up. Useful when some external dependencies are
# not met at build time and break the building process.
autodoc_mock_imports = []

# Controls how to represent typehints.
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# The name for this set of Sphinx documents.
html_title = f"<strong>{project}</strong> <br />{release}"

# -- Patch custom template filters -------------------------------------------


def custom_fullname_filter(fullname):
    return ".".join(fullname.split(".")[1:])


def custom_module_filter(module):
    return module.split(".")[0]


def patch_init(self, app):
    AutosummaryRenderer.__original_init__(self, app)
    self.env.filters["custom_fullname"] = custom_fullname_filter
    self.env.filters["custom_module"] = custom_module_filter


AutosummaryRenderer.__original_init__ = AutosummaryRenderer.__init__
AutosummaryRenderer.__init__ = patch_init
