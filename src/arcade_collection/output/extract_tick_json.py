import json
import tarfile
from typing import Optional, Union

import numpy as np


def extract_tick_json(
    tar: tarfile.TarFile,
    key: str,
    tick: Union[int, float],
    extension: Optional[str] = None,
    field: Optional[str] = None,
) -> list:
    """
    Extract json for specified tick from tar archive.

    For v3 simulations, each tick is saved as a separate json file in the
    archive. The file names are formatted as ``<key>_<tick>.json`` or
    ``<key>_<tick>.<extension>.json`` where tick is padded to six digits. Use an
    integer tick to extract the desired tick from the archive.

    For v2 simulations, all ticks are saved to the same file in the archive. The
    file name is given as ``<key>.json`` or ``<key>.<extension>.json``. Use a
    float tick and field to extract the desired tick from the archive.

    .. code-block:: python

        {
            "timepoints": [
                {
                    "time": <time>, "field": <field>, ...
                }
            ]
        }

    Parameters
    ----------
    tar
        Tar archive.
    key
        Simulation key.
    tick
        Tick to extract.
    extension
        Additional extension in file name.
    field
        Field in json to extract (only used with float ticks).

    Returns
    -------
    :
        Archive contents for specified tick.
    """

    formatted_tick = f"_{tick:06d}" if isinstance(tick, (int, np.integer)) else ""

    if extension is None:
        member = tar.extractfile(f"{key}{formatted_tick}.json")
    else:
        member = tar.extractfile(f"{key}{formatted_tick}.{extension}.json")

    assert member is not None
    tick_json = json.loads(member.read().decode("utf-8"))

    if isinstance(tick, float):
        tick_json = next(item for item in tick_json["timepoints"] if item["time"] == tick)[field]

    return tick_json
