#!/usr/bin/env jupyter
"""
Run two rounds of pycytominer.operations.transform.spherize on profiles and check if it improves retrievability.
Data is pooled by well data
"""
# %% Imports

import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pycytominer.cyto_utils import output
from pycytominer.feature_select import feature_select
from tqdm import tqdm

from correct_position_effect import (regress_out_cell_counts_parallel,
                                     subtract_well_mean_parallel)
from metadata import add_jump_metadata, add_metadata
from methods import (drop_na_inf, feature_select, mad_robustize,
                     remove_outliers, spherize)
from utils import load_profiles_threaded
from wrappers import apply_scaler, scale_grouped_parallel

# %%

sample = None

now = datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
run = Path("./runs/") / now
run.mkdir(parents=True, exist_ok=True)


def simple_save(data: pd.DataFrame, label: str or None = None):
    label = label or "output"

    data.to_parquet(
        path=run / f"{label}.parquet",
        engine="pyarrow",
        compression="zstd",
    )


# %%

# ( scaler, level ) -> kwargs
config = {
    # Alex
    (subtract_well_mean_parallel,): {},
    (regress_out_cell_counts_parallel,): {"cc_col": "Nuclei_Number_Object_Number"},
    # John
    (mad_robustize,): {"epsilon": 1e2},
    (remove_outliers,): {"samples_thresh": 1e4},
    (drop_na_inf,): {"axis": 1},
    # Standard
    (feature_select,): {
        "operation": [
            "variance_threshold",
            "blocklist",
            "drop_outliers",  # Check if this is done in profiling wf
        ]
    },
    (feature_select,): {
        "operation": [
            "noise_removal",
        ],
        "noise_removal_perturb_groups": "Metadata_JCP2022",
        "noise_removal_stdev_cutoff": 1.0,
        "samples": "'Metadata_pert_type'=='trt'",
    },
    (feature_select,): {
        "operation": [
            "correlation_threshold",
        ]
    },
    (spherize,): {"method": "PCA"},
    (spherize, "Metadata_Batch"): {"method": "PCA"},
}

# %%
sources = list(Path("../inputs").rglob("*.parquet"))

# %%

if sample is not None:  # Sample data
    np.random.seed(42)
    sources = np.random.choice(sources, 2, replace=False)

# %%

data = load_profiles_threaded(sources)
data = pd.concat(data, axis=0, ignore_index=True)

# Append metadata for batch-specific scaling
metadata = pd.read_csv("../metadata/experiment_metadata.csv")[
    ["Assay_Plate_Barcode", "Batch_name", "Plate_Map_Name"]
]

plate_to_batch = {
    plate: batch
    for plate, batch in metadata[["Assay_Plate_Barcode", "Batch_name"]].values
}

data["Metadata_Batch"] = data["Metadata_Plate"].map(plate_to_batch)


print("Adding control metadata to dataset")
processed_data = add_jump_metadata(data)

# %% Execution

print(f"Starting pipeline run {now}")

pool = True
for scaler_level, kwargs in tqdm(config.items()):
    scaler, *level = scaler_level
    level = level[0] if len(level) else None

    if level:  # Split into subsets and process them
        processed_data = scale_grouped_parallel(
            data=processed_data, column=level, scaler=scaler, pool=pool, **kwargs
        )
    else:
        processed_data = scaler(processed_data, **kwargs)

    if scaler is spherize:
        simple_save(processed_data, label=f"sphering_{level}")
        pool = False
    elif scaler is feature_select:
        simple_save(processed_data, label="baseline")
