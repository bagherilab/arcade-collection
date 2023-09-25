from typing import Optional


def get_location_voxels(location: dict, region: Optional[str] = None) -> list[tuple[int, int, int]]:
    voxels = [
        (x, y, z)
        for loc in location["location"]
        for x, y, z in loc["voxels"]
        if not region or loc["region"] == region
    ]
    return voxels
