from typing import Tuple, Optional

from prefect import task
import numpy as np
import pandas as pd


@task
def merge_region_samples(
    samples: dict[str, pd.DataFrame], margins: Tuple[int, int, int]
) -> pd.DataFrame:
    default_samples = samples["DEFAULT"]
    all_samples = tranform_sample_coordinates(default_samples, margins)

    regions = [key for key in samples.keys() if key != "DEFAULT"]
    all_region_samples = []

    for region in regions:
        region_samples = tranform_sample_coordinates(samples[region], margins, default_samples)
        region_samples["region"] = region
        all_region_samples.append(region_samples)

    if len(all_region_samples) > 0:
        all_samples = all_samples.merge(
            pd.concat(all_region_samples), on=["id", "x", "y", "z"], how="left"
        )
        all_samples["region"].fillna("DEFAULT", inplace=True)

    valid_samples = filter_valid_samples(all_samples)

    return valid_samples


def tranform_sample_coordinates(
    samples: pd.DataFrame,
    margins: Tuple[int, int, int],
    reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Transforms samples into centered coordinates.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    margins
        Margin size in x, y, and z directions.
    reference
        Reference samples used to calculate transformation.

    Returns
    -------
    :
        Transformed sample cell ids and coordinates.
    """
    if reference is None:
        reference = samples

    minimums = (min(reference.x), min(reference.y), min(reference.z))
    offsets = np.subtract(margins, minimums) + 1

    coordinates = samples[["x", "y", "z"]].values + offsets
    coordinates = coordinates.astype("int64")

    transformed_samples = pd.DataFrame(coordinates, columns=["x", "y", "z"])
    transformed_samples.insert(0, "id", samples["id"])

    return transformed_samples


def filter_valid_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Filters samples for valid cell ids.

    Filter conditions include:

    - Each cell must have at least one sample assigned to each specified region

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
    :
        Valid sample cell ids and coordinates.
    """
    if "region" in samples.columns:
        num_regions = len(samples.region.unique())
        samples = samples.groupby("id").filter(lambda x: len(x.region.unique()) == num_regions)

    return samples.reset_index(drop=True)
