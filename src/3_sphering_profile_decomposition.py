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

# %% [markdown] Goal/Summary
"""
Goal: Find out the reason behind the spikes caused by global and batch sphering

Conclusions:
 - Spikes are not produced during data aggregation, they arise during sphering
"""
# %% Imports

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from process import split_and_process

# %% Setup and data load
run_file = "20231111_175146/sphering_Metadata_Batch.parquet"
profile_dir = Path(f"./runs/{run_file}")
figs_dir = Path("../figs")

kwargs = dict(
    pos_sameby=["Metadata_JCP2022"],
    pos_diffby=[],
    neg_sameby=["Metadata_Plate"],
    neg_diffby=["Metadata_negcon"],
    null_size=10000,
    batch_size=20000,
)

# %% Load data
profiles = pd.read_parquet(profile_dir)
profiles["Metadata_negcon"] = np.where(profiles["Metadata_pert_type"] == "negcon", 1, 0)
# %% Run AP pipeline and aggregate
result = split_and_process(profiles, **kwargs)
aggregation = result.groupby(pos_sameby)["average_precision"].agg(["mean", "count"])
# %% Prepare plot aestheti
# %% Plot ap vs count
sns.scatterplot(data=aggregation, x="mean", y="count")
plt.savefig(figs_dir / "scatter_map_mean_count.png")
plt.close()


#     result["average_precision"]
#     .value_counts()
#     .reset_index()
#     .sort_values("average_precision")
# )

# %% Check which batches compose the spikes

ax = sns.histplot(
    data=result.replace(mapper).sort_values("Metadata_Batch"),
    x="average_precision",
    y="Metadata_Batch",
    hue="Metadata_Batch",
    # multiple="stack",
)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
plt.title("After second sphering")
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig(figs_dir / "histplot_preagg_ap_mean_count.png")
plt.close()

# %% Group by Batch and JCP
batch_jcp_grouped = (
    result.groupby(["Metadata_Batch", "Metadata_JCP2022"])["average_precision"]
    .mean()
    .reset_index()
)
batch_jcp_grouped = batch_jcp_grouped.replace(mapper).sort_values("Metadata_Batch")
# %% Violin plot to show batch

sns.violinplot(
    data=batch_jcp_grouped,
    y="Metadata_Batch",
    x="average_precision",
    hue="Metadata_Batch",
)
plt.savefig(figs_dir / "AP_perBatch.png", dpi=250)
plt.close()

# %% Ongoing work


ax = sns.histplot(
    data=result.replace(mapper).sort_values("Metadata_Batch"),
    x="average_precision",
    y="Metadata_Batch",
    hue="Metadata_Batch",
    # multiple="stack",
)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
plt.title("After second sphering")
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig(figs_dir / "histplot_preagg_ap_mean_count.png")
plt.close()
