"""
Helper functions
"""

import logging

import pandas as pd
from broad_babel.query import run_query

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s:%(message)s")
logging.getLogger("metadata").setLevel(logging.INFO)


def add_metadata(data: pd.DataFrame, platemap_metadata: pd.DataFrame):
    """
    Generate a dataframe with information on perturbation types to be appended to an existing dataframe
    This also homoegeneises control metadata to fit in one column instead of two.
    Inputs:
       data: Data Frame containing features and basic metadata (batch, plate, well)
       Metadata_Plate: Data Frame that maps a plate name to a layout (present as txts)
    Requires:
       - ../metadata/platemaps/{platemaps}.txt exist and contain plate layouts.
       - broad_babel installed (which provides info on what is a control)
    """

    def find_first_return_other_col(
        data: pd.DataFrame, query: str, input_col: str, output_col: str
    ):
        # Use one column in a df to return another
        # MAYBE there is a simpler way using pd.query?
        return data.iloc[data[input_col].eq(query).argmax()][output_col]

    plate_maps = []
    for plate_barcode in data["Metadata_Plate"].unique():
        # platemap_name = platemap_metadata.iloc[
        #     platemap_metadata["Assay_Plate_Barcode"].eq(plate_barcode).argmax()
        # ]["Plate_Map_Name"]
        plate_map_name = find_first_return_other_col(
            platemap_metadata, plate_barcode, "Assay_Plate_Barcode", "Plate_Map_Name"
        )
        plate_map = pd.read_csv(f"../metadata/platemaps/{plate_map_name}.txt", sep="\t")

        # Prioritise found name over file name
        if plate_map_name not in plate_map.columns:
            plate_map["plate_map_name"] = plate_map_name

        plate_maps.append(plate_map)
    platemaps_meta = pd.concat(plate_maps, ignore_index=True)

    def try_query(x: str or float):
        result = ("None", "None")
        if isinstance(x, str):  # skip NaNs
            try:
                result = run_query(
                    x,
                    input_column="broad_sample",
                    output_column="pert_type,control_type",
                )[0]
            except:
                if x.startswith("BRD"):
                    try:  # If it is not found directly and it starts with BRD, try this
                        result = run_query(
                            f"{x[:13]}%",
                            input_column="broad_sample",
                            output_column="pert_type,control_type",
                            operator="LIKE",
                        )[0]
                    except:
                        logging.log(level=logging.ERROR, msg=f"Query {x} failed")
                else:
                    logging.log(level=logging.ERROR, msg=f"Query {x} failed")
        return result

    control_info = platemaps_meta["broad_sample"].map(try_query)

    # Add control information
    for i, col in enumerate(("pert_type", "control_type")):
        platemaps_meta[col] = control_info.map(lambda x: x[i])

    # Homogenise structure to have control type within pert_type column
    control_type = platemaps_meta["control_type"].isin(
        ("poscon_cp", "negcon", "poscon_diverse")
    )
    platemaps_meta.loc[control_type, "pert_type"] = platemaps_meta.loc[
        control_type, "control_type"
    ]

    plate_well_to_perturbation = {
        plate_map: {
            well_pert.well_position: well_pert.pert_type
            for well_pert in platemaps_meta.loc[
                platemaps_meta["plate_map_name"] == plate_map
            ][["well_position", "pert_type"]].itertuples()
        }
        for plate_map in platemaps_meta["plate_map_name"].unique()
    }

    # Copy Plate_Map info from platemap_metadata to data

    data["Metadata_Plate_Map"] = data["Metadata_Plate"].map(
        lambda x: find_first_return_other_col(
            platemap_metadata, x, "Assay_Plate_Barcode", "Plate_Map_Name"
        )
    )
    data["Metadata_control_type"] = [
        plate_well_to_perturbation[plate][well]
        for plate, well in data[["Metadata_Plate_Map", "Metadata_Well"]].to_numpy()
    ]

    return data
