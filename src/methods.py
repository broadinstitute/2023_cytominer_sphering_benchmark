#!/usr/bin/env jupyter
"""
Methods to be combined to generate pipelines
"""
import logging

import numpy as np
import pandas as pd
from pycytominer.feature_select import feature_select
from pycytominer.operations.transform import RobustMAD, Spherize

from cleaning import drop_outlier_feats, drop_outlier_samples
from utils import find_feat_cols
from wrappers import apply_scaler, apply_scaler_on_features

logger = logging.getLogger(__name__)


def drop_na_inf(dframe: pd.DataFrame, axis: int = 0):
    """Drop NaN and Inf values in the features"""
    dframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = find_feat_cols(dframe)
    if axis == 0:
        ignored = False
        for col in feat_cols:
            ignored = dframe[col].isna() | ignored
        num_ignored = ignored.sum()
        if num_ignored == 0:
            return dframe
        dframe = dframe[~ignored]
    else:
        ignored = []
        for col in feat_cols:
            if dframe[col].isna().any():
                ignored.append(col)
        num_ignored = len(ignored)
        if num_ignored == 0:
            return dframe
        dframe = dframe[dframe.columns.difference(ignored)]

    dim = "rows" if axis == 0 else "cols"
    logger.info(f"{num_ignored} deleted {dim} with NaN")
    return dframe


def remove_outliers(
    dframe: pd.DataFrame, feats_thresh: float = 1e2, samples_thresh: float = 1e2
):
    """Remove outliers"""
    dframe, _ = drop_outlier_feats(dframe, threshold=feats_thresh)
    dframe = drop_outlier_samples(dframe, threshold=samples_thresh)
    return dframe


def spherize(data: pd.DataFrame, **kwargs):
    return apply_scaler_on_features(data, Spherize, **kwargs)


def mad_robustize(data: pd.DataFrame, **kwargs):
    return apply_scaler_on_features(data, RobustMAD, **kwargs)
