from typing import Optional

import numpy as np
import pandas as pd


def merge_region_samples(
    samples: dict[str, pd.DataFrame], margins: tuple[int, int, int]
) -> pd.DataFrame:
    """
    Merge different region samples into single valid samples dataframe.

    The input samples are formatted as:

    .. code-block:: python

        {
            "DEFAULT": (dataframe with columns = id, x, y, z),
            "<REGION>": (dataframe with columns = id, x, y, z),
            "<REGION>": (dataframe with columns = id, x, y, z),
            ...
        }

    The DEFAULT region is used as the superset of (x, y, z) samples; any sample
    found only in a non-DEFAULT region are ignored. For a given id, there must
    be at least one sample in each region.

    The output samples are formatted as:

    .. code-block:: markdown

        ┍━━━━━━┯━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━━┑
        │  id  │    x     │    y     │    z     │  region  │
        ┝━━━━━━┿━━━━━━━━━━┿━━━━━━━━━━┿━━━━━━━━━━┿━━━━━━━━━━┥
        │ <id> │ <x + dx> │ <y + dy> │ <z + dz> │ DEFAULT  │
        │ <id> │ <x + dx> │ <y + dy> │ <z + dz> │ <REGION> │
        │ ...  │   ...    │   ...    │   ...    │   ...    │
        │ <id> │ <x + dx> │ <y + dy> │ <z + dz> │ <REGION> │
        ┕━━━━━━┷━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━━┙

    Samples that are found in the DEFAULT region, but not in any non-DEFAULT
    region are marked as DEFAULT. Otherwise, the sample is marked with the
    corresponding region. Region samples should be mutually exclusive.

    Parameters
    ----------
    samples
        Map of region names to region samples.
    margins
        Margin in the x, y, and z directions applied to sample locations.

    Returns
    -------
    :
        Dataframe of merged samples with applied margins.
    """

    default_samples = samples["DEFAULT"]
    all_samples = transform_sample_coordinates(default_samples, margins)

    regions = [key for key in samples.keys() if key != "DEFAULT"]
    all_region_samples = []

    for region in regions:
        region_samples = transform_sample_coordinates(samples[region], margins, default_samples)
        region_samples["region"] = region
        all_region_samples.append(region_samples)

    if len(all_region_samples) > 0:
        all_samples = all_samples.merge(
            pd.concat(all_region_samples), on=["id", "x", "y", "z"], how="left"
        )
        all_samples["region"].fillna("DEFAULT", inplace=True)

    valid_samples = filter_valid_samples(all_samples)

    return valid_samples


def transform_sample_coordinates(
    samples: pd.DataFrame,
    margins: tuple[int, int, int],
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
