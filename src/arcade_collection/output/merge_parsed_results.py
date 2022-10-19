import pandas as pd
from prefect import task


@task
def merge_parsed_results(*results: pd.DataFrame) -> pd.DataFrame:
    for result in results:
        result.set_index(["ID", "TICK"], inplace=True)

    merged_results = pd.concat(results, axis=1)

    return merged_results.reset_index()
