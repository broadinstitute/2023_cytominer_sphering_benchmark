"""
Helper functions
"""

import logging

import numpy as np
import pandas as pd
from broad_babel.query import run_query
from pathos.multiprocessing import Pool

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s:%(message)s")
logging.getLogger("metadata").setLevel(logging.INFO)


def add_metadata(
    data: pd.DataFrame,
    platemap_minimal_meta: pd.DataFrame = None,
    pool: bool or int = True,
    cols_to_add: list = ["Metadata_control_type", "Metadata_broad_samples"],
):
    """
    Generate a dataframe with information on perturbation types to be appended to an existing dataframe
    This also homoegeneises control metadata to fit in one column instead of two.
    Inputs:
       data: Data Frame containing features and basic metadata (batch, plate, well)
       platemap_minimal_meta: Data Frame that maps a plate name to a layout (present as txts)
    Requires:
       - ../metadata/platemaps/{platemaps}.txt exist and contain plate layouts.
       - broad_babel installed (which provides info on what is a control)
    """

    if platemap_minimal_meta is None:
        # FUTURE remove warning once platemap metadata is publicly available
        raise Warning("Experiment metadata is not public yet")
        platemap_minimal_meta = pd.read_csv(
            "https://raw.githubusercontent.com/jump-cellpainting/morphmap/87e6e0c954964b5e80b40a4ffd27a2536ab702d2/00.0.explore-data/output/experiment-metadata.tsv",
            sep="\t",
        )

    if pool:
        if isinstance(pool, bool):
            pool = None
        with Pool(pool) as p:
            plate_maps = p.map(
                lambda x: find_plate_map_file(platemap_minimal_meta, x),
                data["Metadata_Plate"].unique(),
            )
    else:
        plate_maps = [
            find_plate_map_file(platemap_minimal_meta, x)
            for x in data["Metadata_Plate"].unique()
        ]

    platemaps_meta = pd.concat(plate_maps, ignore_index=True)

    unique_ids = platemaps_meta["broad_sample"].unique()

    print("Querying perturbation types")
    with Pool() as p:
        control_info = p.map(try_query, unique_ids)

    sample_control = {k: v for k, v in zip(unique_ids, control_info)}

    tmp = np.array(
        [
            sample_control.get(sample, ("None", "None"))
            for sample in platemaps_meta["broad_sample"]
        ]
    )
    tmp[tmp == None] = "None"
    platemaps_meta[["pert_type", "control_type"]] = tmp

    # Homogenise structure to have control type within pert_type column
    control_type = platemaps_meta["control_type"].isin(
        ("poscon_cp", "negcon", "poscon_diverse")
    )
    platemaps_meta.loc[control_type, "pert_type"] = platemaps_meta.loc[
        control_type, "control_type"
    ]

    id_pert = pd.Series(
        platemaps_meta["pert_type"].values, index=platemaps_meta["broad_sample"]
    ).to_dict()

    print("Find Plate_Map_Name")
    with Pool() as p:
        data["Metadata_Plate_Map"] = p.map(
            lambda x: find_first_return_other_col(
                platemap_minimal_meta,
                x,
                "Assay_Plate_Barcode",
                "Plate_Map_Name",
            ),
            data["Metadata_Plate"],
        )

    print("Assign broad samples")
    plate_well_to_broad_sample = plate_well_to_field(platemaps_meta, "broad_sample")
    with Pool() as p:
        data["Metadata_broad_sample"] = p.map(
            lambda x: plate_well_to_broad_sample[x[0]][x[1]],
            data[["Metadata_Plate_Map", "Metadata_Well"]].to_numpy(),
        )

    with Pool() as p:
        data["Metadata_pert_type"] = p.map(
            lambda x: id_pert.get(x), data["Metadata_broad_sample"]
        )

    return data


def try_query(x: str or float, input_column: str = "broad_sample"):
    result = ("None", "None")
    if isinstance(x, str):  # skip NaNs
        try:
            result = run_query(
                x,
                input_column=input_column,
                output_column="pert_type,control_type",
            )[0]
        except:
            if x.startswith("BRD"):
                try:  # If it is not found directly and it starts with BRD, try this
                    result = run_query(
                        f"{x[:13]}%",
                        input_column=input_column,
                        output_column="pert_type,control_type",
                        operator="LIKE",
                    )[0]
                except:
                    logging.log(level=logging.ERROR, msg=f"Query {x} failed")
            else:
                logging.log(level=logging.ERROR, msg=f"Query {x} failed")
    return result


def find_first_return_other_col(
    data: pd.DataFrame, query: str, input_col: str, output_col: str
) -> str:
    # Use one column in a df to return another
    # MAYBE there is a simpler way using pd.query?
    return data.iloc[data[input_col].eq(query).argmax()][output_col]


def find_plate_map_file(
    platemaps_meta: pd.DataFrame,
    plate_barcode: str,
    commit: str = "6552726ce60a47d3c4c7846fe1766a7c08f96fc1",
) -> pd.DataFrame:
    """Fetch JUMP-ORF plate barcode from public git location and integrate it into an existing data frame.

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
        platemaps_meta, plate_barcode, "Assay_Plate_Barcode", "Plate_Map_Name"
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


def plate_well_to_field(platemaps_meta: pd.DataFrame, field: str):
    """
    Return a dicitonary with structure plate -> well -> field,
    where field is a given column in platemaps_meta.
    """
    return {
        plate_map: {
            well_pert.well_position: getattr(well_pert, field)
            for well_pert in platemaps_meta.loc[
                platemaps_meta["plate_map_name"] == plate_map
            ][["well_position", field]].itertuples()
        }
        for plate_map in platemaps_meta["plate_map_name"].unique()
    }


def add_jump_metadata(
    data: pd.DataFrame,
    ref_field: str = "Metadata_JCP2022",
    pert_field: str = "Metadata_pert_type",
    pool: bool = True,
):
    """Append jump metadata to existing metadata dataframe and remove invalid entries.

    Parameters
    ----------
    data : pd.DataFrame
        Metadata dataframe
    ref_field : str
        Field used for broad_babel queries and to place manually assigned
        entries.
    pert_field : str
        Name of field for perturbations
    pool : bool
        Whether or not to thread broad_babel queries

    Examples
    --------
    FIXME: Add docs.

    """
    wells = pd.read_csv(
        "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz"
    )

    profiles_wmeta = pd.merge(
        data,
        wells,
        on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
    )

    print("Querying perturbation types")
    unique_jcp_ids = profiles_wmeta[ref_field].unique()
    with Pool() as p:
        control_info = np.array(
            p.map(lambda x: try_query(x, "JCP2022"), unique_jcp_ids)
        )
    controls_mask = control_info[:, 0] == "control"
    np.put(control_info[:, 0], controls_mask, control_info[controls_mask, 1])
    combined_control_info = control_info[:, 0]

    # This has to be before renaming JCPS
    mapper_jcp2control = {
        jcp: control for jcp, control in zip(unique_jcp_ids, combined_control_info)
    }

    profiles_wmeta[pert_field] = profiles_wmeta[ref_field].apply(
        lambda x: mapper_jcp2control.get(x, None)
    )

    # Use readable names for controls and non-treatment codes
    # Some compounds do not have control information, nor broad_sample association
    # 'JCP2022_999999', 'JCP2022_050797'
    # These where obtained from John Arevalo's code https://github.com/carpenter-singh-lab/2023_Arevalo_BatchCorrection/blob/95052ea070195f1749e50f9ad058cfa53d7cc430/loader.py#L18()
    MAPPER = {
        "JCP2022_085227": "Aloxistatin",
        "JCP2022_037716": "AMG900",
        "JCP2022_025848": "Dexamethasone",
        "JCP2022_046054": "FK-866",
        "JCP2022_035095": "LY2109761",
        "JCP2022_064022": "NVS-PAK1-1",
        "JCP2022_050797": "Quinidine",
        "JCP2022_012818": "TC-S-7004",
        "JCP2022_033924": "DMSO",
        "JCP2022_999999": "UNTREATED",
        "JCP2022_UNKNOWN": "UNKNOWN",
        "JCP2022_900001": "BAD CONSTRUCT",
    }

    # And assign them as negative controls (Remember to remove invalid entries)
    # Otherwise they will be used as negcons
    manual_negcons = profiles_wmeta[ref_field].isin(MAPPER.keys())
    profiles_wmeta.loc[manual_negcons, pert_field] = "negcon"
    profiles_wmeta[ref_field] = profiles_wmeta[ref_field].apply(
        lambda x: MAPPER.get(x, x)
    )

    # Filter out wells
    profiles_wmeta = profiles_wmeta[
        ~profiles_wmeta[ref_field].isin(["UNTREATED", "UNKNOWN", "BAD CONSTRUCT"])
    ]

    # Consolidate positive controls
    profiles_wmeta[pert_field].replace(
        {k: "poscon" for k in profiles_wmeta[pert_field].unique() if "poscon" in k}
    )
    return profiles_wmeta
