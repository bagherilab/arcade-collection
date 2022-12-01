import json
import tarfile

from prefect import task


@task
def extract_tick_json(tar: tarfile.TarFile, key: str, tick: int, extension: str) -> list[dict]:
    member = tar.extractfile(f"{key}_{tick:06d}.{extension}.json")
    assert member is not None
    tick_json = json.loads(member.read().decode("utf-8"))
    return tick_json
