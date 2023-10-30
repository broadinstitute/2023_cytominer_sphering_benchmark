"""
Helper functions
"""

import logging

import pandas as pd
from broad_babel.query import run_query
from pathos.multiprocessing import Pool

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s:%(message)s")
logging.getLogger("metadata").setLevel(logging.INFO)


def add_metadata(
    data: pd.DataFrame, platemap_metadata: pd.DataFrame = None, pool: bool or int = True
):
    """
    Generate a dataframe with information on perturbation types to be appended to an existing dataframe
    This also homoegeneises control metadata to fit in one column instead of two.
    Inputs:
       data: Data Frame containing features and basic metadata (batch, plate, well)
       platemap_metadata: Data Frame that maps a plate name to a layout (present as txts)
    Requires:
       - ../metadata/platemaps/{platemaps}.txt exist and contain plate layouts.
       - broad_babel installed (which provides info on what is a control)
    """

    if platemap_metadata is None:
        # FUTURE remove warning once platemap metadata is publicly available
        raise Warning("Experiment metadata is not public yet")
        platemap_metadata = pd.read_csv(
            "https://raw.githubusercontent.com/jump-cellpainting/morphmap/87e6e0c954964b5e80b40a4ffd27a2536ab702d2/00.0.explore-data/output/experiment-metadata.tsv",
            sep="\t",
        )

    def find_first_return_other_col(
        data: pd.DataFrame, query: str, input_col: str, output_col: str
    ) -> str:
        # Use one column in a df to return another
        # MAYBE there is a simpler way using pd.query?
        return data.iloc[data[input_col].eq(query).argmax()][output_col]

    def find_plate_map_file(
        plate_barcode: str, commit: str = "6552726ce60a47d3c4c7846fe1766a7c08f96fc1"
    ) -> pd.DataFrame:
        """Fetch plate barcode from public git location.

        Parameters
        ----------
        plate_barcode : str
            name of plate map to fetch
        commit : str
            commit chosen to fetch

        Returns
        -------
        pd.DataFrame
            Dataframe with plate map information, including well info.

        Examples
        --------
        FIXME: Add docs.

        """
        plate_map_name = find_first_return_other_col(
            platemap_metadata, plate_barcode, "Assay_Plate_Barcode", "Plate_Map_Name"
        )
        for batch in (
            "2021_04_26_Batch1",
            "2021_05_10_Batch3",
            "2021_05_17_Batch4",
            "2021_05_31_Batch2",
            "2021_06_07_Batch5",
            "2021_06_14_Batch6",
            "2021_06_21_Batch7",
            "2021_07_12_Batch8",
            "2021_07_26_Batch9",
            "2021_08_02_Batch10",
            "2021_08_09_Batch11",
            "2021_08_23_Batch12",
            "2021_08_30_Batch13",
        ):  # explore known batches
            # plate_map = pd.read_csv(f"../metadata/platemaps/{plate_map_name}.txt", sep="\t")
            try:
                plate_map = pd.read_csv(
                    f"https://raw.githubusercontent.com/jump-cellpainting/jump-orf-data/{commit}/metadata/platemaps/{batch}/platemap/{plate_map_name}.txt",
                    sep="\t",
                )
                break
            except:
                continue

        else:
            logging.log(logging.warning, f"Missing plate map {plate_map_name}")

        # Prioritise found name over file name
        if plate_map_name not in plate_map.columns:
            plate_map["plate_map_name"] = plate_map_name

        return plate_map

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

    with Pool() as p:
        plate_maps = p.map(find_plate_map_file, data["Metadata_Plate"].unique())
    platemaps_meta = pd.concat(plate_maps, ignore_index=True)

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
