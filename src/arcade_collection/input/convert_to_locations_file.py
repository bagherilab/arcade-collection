from typing import List, Dict, Tuple, Optional

from prefect import task
import pandas as pd


@task
def convert_to_locations_file(samples: pd.DataFrame) -> List[Dict]:
    locations: List[Dict] = []
    samples_by_id = samples.groupby("id")

    for i, (_, group) in enumerate(samples_by_id):
        locations.append(convert_to_location(i + 1, group))

    return locations


def convert_to_location(cell_id: int, samples: pd.DataFrame) -> dict:
    """
    Convert samples to ARCADE .LOCATIONS json format.

    Parameters
    ----------
    cell_id
        Unique cell id.
    samples
        Sample cell ids and coordinates.

    Returns
    -------
    :
        Dictionary in ARCADE .LOCATIONS json format.
    """
    center = get_center_voxel(samples)

    if "region" in samples.columns:
        voxels = [
            {"region": region, "voxels": get_location_voxels(samples, region)}
            for region in samples["region"].unique()
        ]
    else:
        voxels = [{"region": "UNDEFINED", "voxels": get_location_voxels(samples)}]

    location = {
        "id": cell_id,
        "center": center,
        "location": voxels,
    }
    return location


def get_center_voxel(samples: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Gets coordinates of center voxel of samples.

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
    center = (center_x, center_y, center_z)
    return center


def get_location_voxels(
    samples: pd.DataFrame, region: Optional[str] = None
) -> List[Tuple[int, int, int]]:
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
