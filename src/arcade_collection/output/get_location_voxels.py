from typing import Optional


def get_location_voxels(location: dict, region: Optional[str] = None) -> list[tuple[int, int, int]]:
    """
    Get list of voxels from location for specified region.

    If region is not given, all voxels in the location are returned, even if
    those voxels are divided into separate regions. If region is given, only
    voxels in that region are returned.

    Parameters
    ----------
    location
        Location object.
    region
        Location region.

    Returns
    -------
    :
        List of x, y, z voxels.
    """

    voxels = [
        (x, y, z)
        for loc in location["location"]
        for x, y, z in loc["voxels"]
        if not region or loc["region"] == region
    ]

    return voxels
