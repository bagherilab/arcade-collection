from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tarfile


def extract_tick_json(
    tar: tarfile.TarFile,
    key: str,
    tick: float,
    extension: str | None = None,
    field: str | None = None,
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

    if member is None:
        message = "File does not exist in archive."
        raise RuntimeError(message)

    tick_json = json.loads(member.read().decode("utf-8"))

    if isinstance(tick, float):
        tick_json = next(item for item in tick_json["timepoints"] if item["time"] == tick)[field]

    return tick_json
