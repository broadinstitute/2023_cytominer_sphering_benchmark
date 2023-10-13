#!/usr/bin/env jupyter
"Wrappers that homogeneise method interfaces."

from typing import Callable

import pandas as pd
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
    fitted_scaler = scaler(**kwargs).fit(data)
    results = fitted_scaler.transform(data)
    if isinstance(results, pd.DataFrame):
        results = results.values
    return pd.DataFrame(
        results,
        # columns=infer_cp_features(data, image_features=False),
        columns=data.columns,
        index=data.index,
    )

    # normalized = meta_df.merge(feature_df, left_index=True, right_index=True)


def apply_scaler_on_features(
    data: pd.DataFrame, scaler: Spherize or RobustMAD, **kwargs
):
    """
    Split features and metadata and then apply a scaler operation to the features. Return the original data frame scaled.
    """
    features, meta = split_meta(data)
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


def scale_grouped_parallel(data: pd.DataFrame, column: str, scaler: Callable, **kwargs):
    """
    Parallelises a scaler on subsets of the data.
    """
    groped_data = [
        processed_data.loc[processed_data[column] == group_id]
        for group_id in processed_data[column].unique()
    ]

    with Pool() as p:
        results = p.map(
            lambda x: apply_scaler_on_features(x, scaler, **kwargs),
            grouped_data,
        )

    return pd.concat(results, axis=0, ignore_index=True)
