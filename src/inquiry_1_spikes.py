#!/usr/bin/env jupyter
"""
Find what correlates to the spike intensity. My candidates are:
- Number of negative controls
- Number of replicates
- Batches
- Plate
"""
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% Imports
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from broad_babel.query import run_query
from process import calculate_map_from_filename

# %% Declare paths
dir_path = Path(
    "/dgx1nas1/storage/data/alan/projects/2023_cytominer_sphering_benchmark/src/runs/20240106_191023"
)

# %% Calculate maps

kwargs = dict(
    pos_sameby=["Metadata_JCP2022"],
    pos_diffby=[],
    neg_sameby=["Metadata_Plate"],
    neg_diffby=["Metadata_negcon"],
    null_size=10000,
    batch_size=20000,
)
maps = {
    prof_path.stem: pl.from_pandas(
        calculate_map_from_filename(prof_path, pool=True, **kwargs)
    )
    for prof_path in dir_path.glob("sphering*parquet")
}

# %% Check their shapes

for k, v in maps.items():
    print(k, v.shape)

# sphering_None (15433, 8)
# sphering_Metadata_Batch (15433, 8)

# %% Recover negative controls using broad-babel


mapper = dict(
    chain.from_iterable(
        [
            run_query(
                query=tuple(maps["sphering_None"]["Metadata_JCP2022"]),
                input_column=input_column,
                output_column=f"{input_column},pert_type",
            )
            for input_column in ("JCP2022", "standard_key")
        ]
    )
)

controls_map = []

for map_values in maps.values():
    double_sph = map_values.with_columns(
        pl.col("Metadata_JCP2022").replace(mapper).alias("pert_type")
    )

    controls_map.append(double_sph.filter(pl.col("pert_type") == "negcon"))

controls = (
    controls_map[0]
    .join(controls_map[1], on="Metadata_JCP2022", suffix="_double")
    .to_pandas()
)

# %% Plot

sns.scatterplot(controls, x="mean_average_precision", y="mean_average_precision_double")
plt.title(f"Controls mAP (n={len(controls)})")
plt.savefig("controls_map_correlation.png", dpi=300)
plt.close()
