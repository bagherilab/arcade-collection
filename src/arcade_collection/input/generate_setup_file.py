from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_POPULATION_ID = "X"
"""Default population ID used in setup file."""


def generate_setup_file(
    samples: pd.DataFrame, margins: tuple[int, int, int], terms: list[str]
) -> str:
    """
    Create ARCADE setup file from samples, margins, and CPM Hamiltonian terms.

    Initial number of cells is determined by number of unique ids in samples.
    Regions are included if samples contains valid regions.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    margins
        Margin size in x, y, and z directions.
    terms
        List of Potts Hamiltonian terms for setup file.

    Returns
    -------
    :
        Contents of ARCADE setup file.
    """

    init = len(samples["id"].unique())
    bounds = calculate_sample_bounds(samples, margins)
    regions = (
        samples["region"].unique()
        if "region" in samples.columns and not samples["region"].isna().all()
        else None
    )
    return make_setup_file(init, bounds, terms, regions)


def calculate_sample_bounds(
    samples: pd.DataFrame, margins: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Calculate transformed sample bounds including margin.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    margins
        Margin size in x, y, and z directions.

    Returns
    -------
    :
        Bounds in x, y, and z directions.
    """

    mins = (min(samples.x), min(samples.y), min(samples.z))
    maxs = (max(samples.x), max(samples.y), max(samples.z))

    bound_x, bound_y, bound_z = np.subtract(maxs, mins) + np.multiply(2, margins) + 3

    return (bound_x, bound_y, bound_z)


def make_setup_file(
    init: int,
    bounds: tuple[int, int, int],
    terms: list[str],
    regions: list[str] | None = None,
) -> str:
    """
    Create ARCADE setup file.

    Parameters
    ----------
    init
        Number of initial cells.
    bounds
        Bounds in x, y, and z directions.
    regions
        List of regions.
    terms
        List of Potts Hamiltonian terms for setup file.

    Returns
    -------
    :
        Contents of ARCADE setup file.
    """

    root = ET.fromstring("<set></set>")  # noqa: S314
    series = ET.SubElement(
        root,
        "series",
        {
            "name": "ARCADE",
            "interval": "1",
            "start": "0",
            "end": "0",
            "dt": "1",
            "ds": "1",
            "ticks": "1",
            "length": str(int(bounds[0])),
            "width": str(int(bounds[1])),
            "height": str(int(bounds[2])),
        },
    )

    potts = ET.SubElement(series, "potts")
    for term in terms:
        ET.SubElement(potts, "potts.term", {"id": term})

    agents = ET.SubElement(series, "agents")
    populations = ET.SubElement(agents, "populations")
    population = ET.SubElement(populations, "population", {"id": "X", "init": str(init)})

    if regions is not None:
        for region in regions:
            ET.SubElement(population, "population.region", {"id": region})

    ET.indent(root, space="    ", level=0)
    return ET.tostring(root, encoding="unicode")
