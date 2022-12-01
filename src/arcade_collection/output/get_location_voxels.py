from typing import Optional

from prefect import task


@task
def get_location_voxels(location: dict, region: Optional[str] = None) -> list[tuple[int, int, int]]:
    voxels = [
        voxel
        for loc in location["location"]
        for voxel in loc["voxels"]
        if not region or loc["region"] == region
    ]
    return voxels
