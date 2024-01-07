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
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from process import calculate_map_from_filename

# %%

sns.set_style("white")
sns.set_palette("Set2")

# %%

logger = logging.getLogger(__name__)

# %% [markdown]
# In this notebook mAP values and the associated p values are computed to estimate ORF replicability.
# Three datasets are compared, baseline (no sphering), global sphering and global+batch sphering
# %% Parameters

pool_map_calculation = True  # Whether or not to run all three datasets in parallel
run = "20231111_175146"


# Alternative way to calculate map
# pos_sameby = ["Metadata_broad_sample", "Metadata_Well"]
kwargs = dict(
    pos_sameby=["Metadata_JCP2022"],
    pos_diffby=[],
    neg_sameby=["Metadata_Plate"],
    neg_diffby=["Metadata_negcon"],
    null_size=10000,
    batch_size=20000,
)

# %% Standard

profiles_dir = Path(f"./runs/{run}")
output_dir = Path("./output")
figs_dir = Path("../figs")
figs_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

filenames = list(profiles_dir.glob("*"))
# type_set for type_set in profiles_dir.glob("*") if "Batch" in str(type_set)


assert filenames and all(
    (map(lambda x: x.exists(), filenames))
), "Some files are misisng"


# %% Process metadata


# %%

aggregated_data = pd.concat(
    [calculate_map_from_filename(filename, **kwargs) for filename in tqdm(filenames)],
    ignore_index=True,
)


aggregated_data.to_csv(output_dir / "replicate_mAP_all_pipelines.csv.gz", index=False)

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
plt.xlabel("mAP (neg_sameby = Metadata_Plate)")
sns.move_legend(ax, loc="upper left")

plt.savefig(figs_dir / f"{run}_all_pipelines.png", dpi=300)
plt.close()
