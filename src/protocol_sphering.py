#!/usr/bin/env jupyter
"""
Run two rounds of pycytominer.operations.transform.spherize on profiles and check if it improves retrievability.
Data is pooled by well data
"""
# %% Imports

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features, load_profiles, output
from pycytominer.feature_select import feature_select
from pycytominer.operations.transform import RobustMAD, Spherize

from cleaning import drop_na_inf, drop_outlier_feats, drop_outlier_samples
from correct_position_effect import (regress_out_cell_counts_parallel,
                                     subtract_well_mean_parallel)

# %%


def remove_outliers(dframe: pd.DataFrame):
    """Remove outliers"""
    dframe, _ = drop_outlier_feats(dframe, threshold=1e2)
    dframe = drop_outlier_samples(dframe, threshold=1e2)
    # dframe = isolation_removal(dframe)
    return dframe


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


def split_meta(data: pd.DataFrame):
    """
    Returns Metadata and data as separate dataframes.
    """
    features, meta = [
        infer_cp_features(data, metadata=x, image_features=True) for x in (False, True)
    ]
    return data.loc(axis=1)[features], data.loc(axis=1)[meta]


def apply_scaler_on_features(
    data: pd.DataFrame, scaler: Spherize or RobustMAD, **kwargs
):
    """
    Split features and metadata and then apply a scaler operation to the features. Return the original data frame scaled.
    """
    features, meta = split_meta(data)
    processed_features = apply_scaler(features, scaler, **kwargs)

    return meta.merge(processed_features, left_index=True, right_index=True)


# %%

sources = list(Path("../inputs").rglob("*.parquet"))

# %%

np.random.seed(42)
samples = np.random.choice(sources, 5, replace=False)
# samples = sources

# %%

sample_data = pd.concat(
    [load_profiles(sample) for sample in samples], axis=0, ignore_index=True
)


# %%


config = {
    "mean_centering": True,
    "cell_count_adjustment": True,
    "mad_robustize": True,
    "remove_outliers": True,
    "remove_nans": True,
    "select_features": True,
    "sphering": True,
    "sphering_2": True,  # Batch-based sphering
}

processed_data = sample_data
if config["mean_centering"]:
    processed_data = subtract_well_mean_parallel(processed_data)
if config["cell_count_adjustment"]:
    processed_data = regress_out_cell_counts_parallel(
        processed_data, cc_col="Nuclei_Number_Object_Number"
    )
if config["mad_robustize"]:
    processed_data = apply_scaler_on_features(processed_data, RobustMAD, epsilon=0)
if config["remove_outliers"]:
    processed_data = remove_outliers(processed_data)
if config["remove_nans"]:
    processed_data = drop_na_inf(processed_data, axis=1)
if config["select_features"]:
    processed_data = feature_select(processed_data)
if config["sphering"]:
    processed_data = apply_scaler_on_features(processed_data, Spherize, method="PCA")
if config["sphering_2"]:
    # Append metadata

    metadata = pd.read_csv("../metadata/experiment_metadata.csv")[
        ["Assay_Plate_Barcode", "Batch_name"]
    ]
    plate_to_batch = {plate: batch for plate, batch in metadata.values}
    processed_data["Metadata_Batch"] = processed_data["Metadata_Plate"].map(
        plate_to_batch
    )
    processed_data = pd.concat(
        [
            apply_scaler_on_features(
                processed_data.loc[processed_data["Metadata_Batch"] == batch],
                Spherize,
                method="PCA",
            )
            for batch in processed_data["Metadata_Batch"].unique()
        ]
    )

now = datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
run = Path("../runs/") / now
run.mkdir(parents=True, exist_ok=True)

output(
    df=processed_data,
    output_filename=run / "processed_data.csv.gz",
    compression_options={"method": "gzip"},
    float_format=None,
)
