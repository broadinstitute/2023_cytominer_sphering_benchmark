#!/usr/bin/env jupyter
"Wrappers that homogeneise method interfaces."

from inspect import isclass
from time import perf_counter
from typing import Callable

import pandas as pd
from pathos.multiprocessing import Pool
from pycytominer.cyto_utils import infer_cp_features
from pycytominer.operations.transform import RobustMAD, Spherize


def apply_scaler(data: pd.DataFrame, scaler: Spherize or RobustMAD, **kwargs):
    """
    Apply any scaler to a Data Frame.
    Reimplement a subset of pytytominer.normalize.

    scaler: BaseEstimator
    data: pd.DataFrame composed of exclusively numeric columns

    See more details on
    https://github.com/cytomining/pycytominer/blob/main/pycytominer/normalize.py
    """
    data_to_fit = kwargs.pop("data_to_fit", data)

    if isclass(scaler):  # Unwrapped method
        fitted_scaler = scaler(**kwargs).fit(data_to_fit)
        results = fitted_scaler.transform(data)
    else:  # Already wrapped method
        results = scaler(data, **kwargs)

    if isinstance(results, pd.DataFrame):
        results = results.values
    return pd.DataFrame(
        results,
        columns=data.columns,
        index=data.index,
    )


def apply_scaler_on_features(
    data: pd.DataFrame, scaler: Callable, fit_negcon: bool = False, **kwargs
):
    """
    Split features and metadata and then apply a scaler operation to the features.
    Optionally, pass negative controls (negcon) separately for fitting.
    Return the original data frame scaled.
    """
    features, meta = split_meta(data)

    if fit_negcon:
        negcon_indices = meta["Metadata_pert_type"] == "negcon"
        negcons = features.loc[negcon_indices]
        assert len(
            negcons
        ), f"No negative controls during step {scaler}. Pipeline interrupted."

        kwargs["data_to_fit"] = negcons

    processed_features = apply_scaler(features, scaler, **kwargs)

    return meta.merge(processed_features, left_index=True, right_index=True)


def split_meta(data: pd.DataFrame):
    """
    Returns Metadata and data as separate dataframes.
    """
    features, meta = [
        infer_cp_features(data, metadata=x, image_features=False) for x in (False, True)
    ]
    return data.loc(axis=1)[features], data.loc(axis=1)[meta]


def scale_grouped_parallel(
    data: pd.DataFrame, column: str, scaler: Callable, pool=True, **kwargs
):
    """
    Parallelises a scaler on subsets of the data.
    """
    grouped_data = [
        data.loc[data[column] == group_id] for group_id in data[column].unique()
    ]

    if pool:
        with Pool() as p:
            results = p.map(
                lambda x: apply_scaler(x, scaler, **kwargs),
                grouped_data,
            )

    else:
        results = [apply_scaler(x, scaler, **kwargs) for x in grouped_data]

    return pd.concat(results, axis=0, ignore_index=True)
