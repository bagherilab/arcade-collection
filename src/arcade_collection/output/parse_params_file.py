from prefect import task
import json
import tarfile
import pandas as pd


@task
def parse_params_file(tar: tarfile.TarFile) -> pd.DataFrame:
    all_timepoints = []

    for member in tar.getmembers():
        seed = 0
        extracted_member = tar.extractfile(member)
        assert extracted_member is not None
        extracted_json = json.loads(extracted_member.read().decode("utf-8"))

        timepoints = [parse_timepoint(timepoint) for timepoint in extracted_json]
        all_timepoints = all_timepoints + timepoints

    timepoints_df = pd.DataFrame(all_timepoints, columns=COLUMN_NAMES)

    return timepoints_df


def parse_timepoint(timepoint):
    return 0
