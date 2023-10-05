#!/usr/bin/env jupyter
"""
Run two rounds of pycytominer.operations.transfor.spherize on profiles and check if it improves retrievability.
"""
# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd
from pycytominer import feature_select, normalize
from pycytominer.feature_select import feature_select
from pycytominer.operations.transform import Spherize

from cleaning import drop_na_inf, drop_outlier_feats, drop_outlier_samples

# %%


sources = list(Path("../inputs").rglob("*.parquet"))

# %%

np.random.seed(42)
samples = np.random.choice(sources, 5, replace=False)

# %%

sample_data = pd.concat([pd.read_parquet(sample) for sample in samples], axis=0)

# outlier removal
# nan removal
# feature selection
# add sphering


def remove_outliers(dframe: pd.DataFrame):
    """Remove outliers"""
    dframe, _ = drop_outlier_feats(dframe, threshold=1e2)
    dframe = drop_outlier_samples(dframe, threshold=1e2)
    # dframe = isolation_removal(dframe)
    return dframe


config = {
    "mad_robustize": True,
    "remove_outliers": True,
    "remove_nans": True,
    "select_features": True,
    "sphering": True,
}

processed_data = sample_data
if config["pcm_norm"]:
    pcm_norm_output = normalize(
        profiles=processed_data,
        # features="infer",
        # image_features=False,
        meta_features="infer",
        # samples="all",
        method="mad_robustize",
        mad_robustize_epsilon=0,
    )

if config["remove_outliers"]:
    processed_data = remove_outliers(processed_data)
if config["remove_nans"]:
    processed_data = drop_na_inf(processed_data)
if config["select_features"]:
    processed_data = feature_select(processed_data)
if config["sphering"]:
    second_sphere = Spherize(processed_data)
