import pandas as pd


def convert_to_cells_file(
    samples: pd.DataFrame,
    reference: pd.DataFrame,
    volume_distributions: dict[str, tuple[float, float]],
    height_distributions: dict[str, tuple[float, float]],
    critical_volume_distributions: dict[str, tuple[float, float]],
    critical_height_distributions: dict[str, tuple[float, float]],
    state_thresholds: dict[str, float],
) -> list[dict]:
    """
    Convert all samples to cell objects.

    For each cell id in samples, current volume and height are rescaled to
    critical volume and critical height based on distribution means and standard
    deviations. If reference volume and/or height exist for the cell id, those
    values are used as the current values to be rescaled. Otherwise, current
    volume is calculated from the number of voxel samples and current height is
    calculated from the range of voxel coordinates along the z axis.

    Initial cell state and cell state phase for each cell are estimated based on
    state thresholds, the current cell volume, and the critical cell volume.

    Cell object ids are reindexed starting with cell id 1.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    reference
        Reference values for volumes and heights.
    volume_distributions
        Map of volume means and standard deviations.
    height_distributions
        Map of height means and standard deviations.
    critical_volume_distributions
        Map of critical volume means and standard deviations.
    critical_height_distributions
        Map of critical height means and standard deviations.
    state_thresholds
        Critical volume fractions defining threshold between states.

    Returns
    -------
    :
        List of cell objects formatted for ARCADE.
    """

    cells: list[dict] = []
    samples_by_id = samples.groupby("id")

    for i, (cell_id, group) in enumerate(samples_by_id):
        cell_reference = filter_cell_reference(cell_id, reference)
        cells.append(
            convert_to_cell(
                i + 1,
                group,
                cell_reference,
                volume_distributions,
                height_distributions,
                critical_volume_distributions,
                critical_height_distributions,
                state_thresholds,
            )
        )

    return cells


def convert_to_cell(
    cell_id: int,
    samples: pd.DataFrame,
    reference: dict,
    volume_distributions: dict[str, tuple[float, float]],
    height_distributions: dict[str, tuple[float, float]],
    critical_volume_distributions: dict[str, tuple[float, float]],
    critical_height_distributions: dict[str, tuple[float, float]],
    state_thresholds: dict[str, float],
) -> dict:
    """
    Convert samples to cell object.

    Current volume and height are rescaled to critical volume and critical
    height based on distribution means and standard deviations. If reference
    volume and/or height are provided (under the "DEFAULT" key), those values
    are used as the current values to be rescaled. Otherwise, current volume is
    calculated from the number of voxel samples and current height is calculated
    from the range of voxel coordinates along the z axis.

    Initial cell state and cell state phase are estimated based on state
    thresholds, the current cell volume, and the critical cell volume.

    Parameters
    ----------
    cell_id
        Unique cell id.
    samples
        Sample coordinates for a single object.
    reference
        Reference data for cell.
    volume_distributions
        Map of volume means and standard deviations.
    height_distributions
        Map of height means and standard deviations.
    critical_volume_distributions
        Map of critical volume means and standard deviations.
    critical_height_distributions
        Map of critical height means and standard deviations.
    state_thresholds
        Critical volume fractions defining threshold between states.

    Returns
    -------
    :
        Cell object formatted for ARCADE.
    """

    volume = len(samples)
    height = samples.z.max() - samples.z.min()

    critical_volume = convert_value_distribution(
        (reference["volume"] if "volume" in reference else volume),
        volume_distributions["DEFAULT"],
        critical_volume_distributions["DEFAULT"],
    )

    critical_height = convert_value_distribution(
        (reference["height"] if "height" in reference else height),
        height_distributions["DEFAULT"],
        critical_height_distributions["DEFAULT"],
    )

    state = get_cell_state(volume, critical_volume, state_thresholds)

    cell = {
        "id": cell_id,
        "parent": 0,
        "pop": 1,
        "age": 0,
        "divisions": 0,
        "state": state.split("_")[0],
        "phase": state,
        "voxels": volume,
        "criticals": [critical_volume, critical_height],
    }

    if "region" in samples.columns and not samples["region"].isnull().all():
        regions = [
            convert_to_cell_region(
                region,
                region_samples,
                reference,
                volume_distributions,
                height_distributions,
                critical_volume_distributions,
                critical_height_distributions,
            )
            for region, region_samples in samples.groupby("region")
        ]
        cell.update({"regions": regions})

    return cell


def convert_to_cell_region(
    region: str,
    region_samples: pd.DataFrame,
    reference: dict,
    volume_distributions: dict[str, tuple[float, float]],
    height_distributions: dict[str, tuple[float, float]],
    critical_volume_distributions: dict[str, tuple[float, float]],
    critical_height_distributions: dict[str, tuple[float, float]],
) -> dict:
    """
    Convert region samples to cell region object.

    Current region volume and height are rescaled to critical volume and
    critical height based on distribution means and standard deviations. If
    reference region volume and/or height are provided, those values are used as
    the current values to be rescaled. Otherwise, current region volume is
    calculated from the number of voxel samples and current region height is
    calculated from the range of voxel coordinates along the z axis.

    Parameters
    ----------
    region
        Region name.
    region_samples
        Sample coordinates for region of a single object.
    reference
        Reference data for cell region.
    volume_distributions
        Map of volume means and standard deviations.
    height_distributions
        Map of height means and standard deviations.
    critical_volume_distributions
        Map of critical volume means and standard deviations.
    critical_height_distributions
        Map of critical height means and standard deviations.

    Returns
    -------
    :
        Cell region object formatted for ARCADE.
    """

    region_volume = len(region_samples)
    region_height = region_samples.z.max() - region_samples.z.min()

    region_critical_volume = convert_value_distribution(
        (reference[f"volume.{region}"] if f"volume.{region}" in reference else region_volume),
        volume_distributions[region],
        critical_volume_distributions[region],
    )

    region_critical_height = convert_value_distribution(
        (reference[f"height.{region}"] if f"height.{region}" in reference else region_height),
        height_distributions[region],
        critical_height_distributions[region],
    )

    return {
        "region": region,
        "voxels": len(region_samples),
        "criticals": [region_critical_volume, region_critical_height],
    }


def get_cell_state(
    volume: float,
    critical_volume: float,
    threshold_fractions: dict[str, float],
) -> str:
    """
    Estimates cell state based on cell volume.

    The threshold fractions dictionary defines the monotonic thresholds between
    different cell states. For a given volume v, critical volume V, and states
    X1, X2, ..., XN with corresponding, monotonic threshold fractions f1, f2,
    ..., fN, a cell is assigned state Xi such that [f(i - 1) * V] <= v < [fi *
    V].

    Cells with v < f1 * V are assigned state X1.

    Cells with v > fN * V are assigned state XN.

    Parameters
    ----------
    volume
        Current cell volume.
    critical_volume
        Critical cell volume.
    threshold_fractions
        Critical volume fractions defining threshold between states.

    Returns
    -------
    :
        Cell state.
    """

    thresholds = [fraction * critical_volume for fraction in threshold_fractions.values()]
    states = list(threshold_fractions.keys())

    index = next((ind for ind, thresh in enumerate(thresholds) if thresh > volume), -1)
    return states[index]


def convert_value_distribution(
    value: float,
    source_distribution: tuple[float, float],
    target_distribution: tuple[float, float],
) -> float:
    """
    Estimates target value based on source value and source and target distributions.

    Parameters
    ----------
    value
        Source value.
    source_distribution
        Average and standard deviation of source value distribution.
    target_distribution
        Average and standard deviation of target value distribution.

    Returns
    -------
    :
        Estimated critical value.
    """

    source_avg, source_std = source_distribution
    target_avg, target_std = target_distribution
    z_scored_value = (value - source_avg) / source_std
    converted_value = z_scored_value * target_std + target_avg
    return converted_value


def filter_cell_reference(cell_id: int, reference: pd.DataFrame) -> dict:
    """
    Filters reference data for given cell id.

    Parameters
    ----------
    cell_id
        Unique cell id.
    reference
        Reference data for conversion.

    Returns
    -------
    :
        Reference data for given cell id.
    """

    cell_reference = reference[reference["ID"] == cell_id].squeeze()
    cell_reference = cell_reference.to_dict() if not cell_reference.empty else {}
    return cell_reference
