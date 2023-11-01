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
