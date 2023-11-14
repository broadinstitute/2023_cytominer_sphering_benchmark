#!/usr/bin/env jupyter
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

# %% [markdown]
"""
Performing a second sphering (Whitening) step discretises the data,
and any sphering increases average mAP as well as the fraction of entries
with low q-values (False Discovery Rate IIRC).

Other things learning during development are:
- parquet files offer much better IO speed at barely any space cost when compared to csv.gz.
- Threading speeds up processing times, so always use it in embarassingly parallel situations.
- Keep the "final" metadata all in one place, ideally an accessible package.

Finally, things to bring up for discussion
- There are many inconsistencies with JUMP metadata; let's put an accessibility tool in a single place.
- Does it makes sense for double-sphered samples to bin the way they are?
  - The number of unique mAP values for each pipeline is: baseline
    - Global + Batch sphering   1126
    - Global sphering           7794
    - No sphering               7432
"""
# # %%
#
#
#

# df = (
#     data.loc(axis=1)[
#         [
#             "Metadata_Plate",
#             "Metadata_Well",
#             "Metadata_pert_type",
#         ]
#     ]
#     .assign(row=lambda x: x.Metadata_Well.str[0:1])
#     .assign(col=lambda x: x.Metadata_Well.str[1:])
# )

# df.columns = [
#     "Metadata_Plate",
#     "well_position",
#     "pert_type",
#     "row",
#     "col",
# ]
# %%
import logging
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copairs.map import aggregate, run_pipeline
from pathos.multiprocessing import Pool
from pycytominer.cyto_utils import load_profiles
from tqdm import tqdm

from utils import get_featuredata, get_metadata, load_profiles_threaded

# %%

sns.set_style("white")
sns.set_palette("Set2")

# %%

logger = logging.getLogger(__name__)

# %% [markdown]
# In this notebook mAP values and the associated p values are computed to estimate ORF replicability.
# Three datasets are compared, baseline (no sphering), global sphering and global+batch sphering
# %% [markdown]
# #### Read the data
# %%

# run = "20231107_153027" # mini version
# run = "20231107_153349" #gz version
# run = "20231108_194207"  # parquet - zstd version
run = "20231111_175146"  # parquet - zstd version
profiles_dir = Path(f"./runs/{run}")
output_dir = Path("./output")
figs_dir = Path("../figs")
figs_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

filenames = [type_set for type_set in profiles_dir.glob("*")]

assert filenames and all(
    (map(lambda x: x.exists(), filenames))
), "Some files are misisng"


def calculate_map_from_filename(filename):
    # You need >12Gb RAM to load them in parallel
    # These profiles already contain metadata about controls
    # data = load_profiles_threaded(filenames)
    # for i, filename in enumerate(filenames) :
    #     data[i]["Metadata_Pipeline"] = filename.name.split(".")[0]

    print("Loading data")
    t = perf_counter()
    data = load_profiles(filename)
    print(f"Data loaded in {perf_counter()-t} seconds")
    data["Metadata_Pipeline"] = filename.name.split(".")[0]

    # data = pd.concat(data, axis=0)

    data["Metadata_negcon"] = np.where(data["Metadata_pert_type"] == "negcon", 1, 0)

    # Alternative way to calculate map
    # pos_sameby = ["Metadata_broad_sample", "Metadata_Well"]
    pos_sameby = ["Metadata_JCP2022"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_negcon"]

    # %% Process metadata
    def run_pipeline_and_aggregate(data, pipeline_type) -> pd.DataFrame:
        data = data.loc[data["Metadata_Pipeline"] == pipeline_type]
        metadata_df = get_metadata(data)
        feature_df = get_featuredata(data)

        print("Running pipeline")
        t = perf_counter()
        result = run_pipeline(
            metadata_df,
            feature_df.values,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            20000,
        )
        print(f"Pipeline run in {perf_counter()-t} seconds")

        aggregated = aggregate(result, pos_sameby, threshold=0.05).rename(
            columns={"average_precision": "mean_average_precision"}
        )
        aggregated["Metadata_Pipeline"] = pipeline_type

        return aggregated

    pool_map_calculation = False
    if pool_map_calculation:
        with Pool() as p:
            aggregated_data = pd.concat(
                p.map(
                    lambda x: run_pipeline_and_aggregate(data, x),
                    data["Metadata_Pipeline"].unique(),
                ),
                ignore_index=True,
            )
    else:
        aggregated_data = pd.concat(
            [
                run_pipeline_and_aggregate(data, x)
                for x in data["Metadata_Pipeline"].unique()
            ]
        )
    return aggregated_data


# %%


aggregated_data = pd.concat(
    [calculate_map_from_filename(filename) for filename in tqdm(filenames)],
    ignore_index=True,
)
aggregated_data.to_csv(output_dir / f"replicate_mAP_all_pipelines.csv.gz", index=False)

# %% Number of compounds for which q<0.05 and plot

recovered_fraction = aggregated_data.groupby("Metadata_Pipeline").apply(
    lambda x: round((sum(x["q_value"] < 0.05) / len(aggregated_data)), 2)
)
pp_mapping = {
    "baseline": f"No sphering ({recovered_fraction.baseline})",
    "sphering_None": f"Global sphering ({recovered_fraction.sphering_None})",
    "sphering_Metadata_Batch": f"Global + Batch sphering ({recovered_fraction.sphering_Metadata_Batch})",
}
# recovered_fraction = f"Average values where q<0.05: { recovered_fraction }"
legend_label = "Pipeline (Frac. q < 0.05)"
aggregated_data[legend_label] = aggregated_data["Metadata_Pipeline"].map(pp_mapping)
ax = sns.histplot(
    data=aggregated_data.sort_values(legend_label),
    x="mean_average_precision",
    hue=legend_label,
    element="step",
    stat="count",
    common_norm=False,
    bins=30,
)
plt.title(
    f"Effect of sphering rounds on technical retrievability (n={int(aggregated_data[legend_label].value_counts().iloc[0])})"
)
# plt.xlabel(f"mAP (neg_sameby = {neg_sameby[0].split('_')[-1]})")
plt.xlabel(f"mAP (neg_sameby = Metadata_Plate)")
sns.move_legend(ax, loc="upper left")

plt.savefig(figs_dir / f"{run}_all_pipelines.png", dpi=300)
plt.close()
