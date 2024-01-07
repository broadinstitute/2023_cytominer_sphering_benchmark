#!/usr/bin/env jupyter
from time import perf_counter

import numpy as np
import pandas as pd
from copairs.map import aggregate, run_pipeline
from pathos.multiprocessing import Pool
from pycytominer.cyto_utils import load_profiles

from utils import get_featuredata, get_metadata


def split_and_process(data: pd.DataFrame, **kwargs):
    metadata_df = get_metadata(data)
    feature_df = get_featuredata(data)

    print("Running pipeline")
    result = run_pipeline(metadata_df, feature_df.values, **kwargs)
    return result


def run_pipeline_and_aggregate(data, pipeline_type, **kwargs) -> pd.DataFrame:
    data = data.loc[data["Metadata_Pipeline"] == pipeline_type]
    print(f"Pipeline run in {perf_counter()-t} seconds")

    result = split_and_process(data)
    aggregated = aggregate(
        result, kwargs["pos_sameby"], threshold=kwargs.get("threhold", 0.05)
    ).rename(columns={"average_precision": "mean_average_precision"})
    aggregated["Metadata_Pipeline"] = pipeline_type

    return aggregated


def calculate_map_from_filename(filename, **kwargs):
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

    data["Metadata_negcon"] = np.where(data["Metadata_pert_type"] == "negcon", 1, 0)

    if kwargs.get("pool_map_calculation", True):
        with Pool() as p:
            aggregated_data = pd.concat(
                p.map(
                    lambda x: run_pipeline_and_aggregate(data, x, **kwargs),
                    data["Metadata_Pipeline"].unique(),
                ),
                ignore_index=True,
            )
    else:
        aggregated_data = pd.concat(
            [
                run_pipeline_and_aggregate(data, x, **kwargs)
                for x in data["Metadata_Pipeline"].unique()
            ]
        )
    return aggregated_data
