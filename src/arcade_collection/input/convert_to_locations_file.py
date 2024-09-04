from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def convert_to_locations_file(samples: pd.DataFrame) -> list[dict]:
    """
    Convert all samples to location objects.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
    :
        List of location objects formatted for ARCADE.
    """

    locations: list[dict] = []
    samples_by_id = samples.groupby("id")

    for i, (_, group) in enumerate(samples_by_id):
        locations.append(convert_to_location(i + 1, group))

    return locations


def convert_to_location(cell_id: int, samples: pd.DataFrame) -> dict:
    """
    Convert samples to location object.

    Parameters
    ----------
    cell_id
        Unique cell id.
    samples
        Sample coordinates for a single object.

    Returns
    -------
    :
        Location object formatted for ARCADE.
    """

    center = get_center_voxel(samples)

    if "region" in samples.columns and not samples["region"].isna().all():
        voxels = [
            {"region": region, "voxels": get_location_voxels(samples, region)}
            for region in samples["region"].unique()
        ]
    else:
        voxels = [{"region": "UNDEFINED", "voxels": get_location_voxels(samples)}]

    return {
        "id": cell_id,
        "center": center,
        "location": voxels,
    }


def get_center_voxel(samples: pd.DataFrame) -> tuple[int, int, int]:
    """
    Get coordinates of center voxel of samples.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
    :
        Center voxel.
    """

    center_x = int(samples["x"].mean())
    center_y = int(samples["y"].mean())
    center_z = int(samples["z"].mean())
    return (center_x, center_y, center_z)


def get_location_voxels(
    samples: pd.DataFrame, region: str | None = None
) -> list[tuple[int, int, int]]:
    """
    Get list of voxel coordinates from samples dataframe.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    region
        Region key.

    Returns
    -------
    :
        List of voxel coordinates.
    """

    if region is not None:
        region_samples = samples[samples["region"] == region]
        voxels_x = region_samples["x"]
        voxels_y = region_samples["y"]
        voxels_z = region_samples["z"]
    else:
        voxels_x = samples["x"]
        voxels_y = samples["y"]
        voxels_z = samples["z"]

    return list(zip(voxels_x, voxels_y, voxels_z))
