import os
import zipfile
import pandas as pd


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a zip file only if the target directory does not exist."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)


def load_csv_from_zip(zip_path: str, extract_dir: str, csv_name: str, cols=None) -> pd.DataFrame:
    """
    Extract a zip and load a CSV from it.
    """
    if csv_name and os.path.exists(os.path.join(extract_dir, csv_name)):
        print(f"Loading existing {csv_name} CSV directly!")
        return pd.read_csv(os.path.join(extract_dir, csv_name), low_memory=False, usecols=cols)

    extract_zip(zip_path, extract_dir)

    csv_path = os.path.join(extract_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
    
    return pd.read_csv(csv_path, low_memory=False, usecols=cols)


def load_imaging_data(data_dir: str, modality: str = "mri") -> pd.DataFrame:
    """
    Load one imaging metadata table.

    modality options:
    - 'mri'
    - 'fdg_pet'
    - 'amyloid_pet'
    - 'tau_pet'
    """
    # Set the dirs where the data is found
    mri_zip = os.path.join(data_dir, "NACC_mri_data.zip")
    pet_zip = os.path.join(data_dir, "NACC_pet_data.zip")

    mri_extract_dir = os.path.join(data_dir, "NACC_mri_data")
    pet_extract_dir = os.path.join(data_dir, "NACC_pet_data")

    file_map = {
        "mri": (mri_zip, mri_extract_dir, "investigator_scan_mrisbm_nacc72.csv"),
        "fdg_pet": (pet_zip, pet_extract_dir, "investigator_scan_fdgpetnpdka_nacc72.csv"),
        "amyloid_pet": (pet_zip, pet_extract_dir, "investigator_scan_amyloidpetnpdka_nacc72.csv"),
        "tau_pet": (pet_zip, pet_extract_dir, "investigator_scan_taupetnpdka_nacc72.csv"),
    }

    zip_path, extract_dir, csv_name = file_map[modality]
    return load_csv_from_zip(zip_path, extract_dir, csv_name)


def load_visitation_data(
    data_dir: str,
    zip_name: str = "NACC_visitation_data.zip",
    csv_name: str = "investigator_ftldlbd_nacc72.csv"
) -> pd.DataFrame:
    """
    Load NACC visitation data.

    If csv_name is not provided, the first CSV found inside the zip is loaded.
    """
    zip_path = os.path.join(data_dir, zip_name)
    extract_dir = os.path.join(data_dir, os.path.splitext(zip_name)[0])
    
    # Columns that need to be loaded
    # This is where we can control what "target col" metric for the mlp
    cols = ["NACCID", "VISITYR", "VISITMO", "VISITDAY", "NACCMMSE", "CDRSUM", "MEMORY", "CDRLANG", "DEMENTED"]
    
    return load_csv_from_zip(zip_path, extract_dir, csv_name, cols)


def prepare_imaging_df(imaging_df: pd.DataFrame, modality: str) -> pd.DataFrame:
    print("\n=== PREPARING IMAGING DATA ===")
    # print("Original imaging shape:", imaging_df.shape)
    # print("Imaging columns:")
    # print(imaging_df.columns.tolist())
    
    # ID
    ID_COL = "NACCID"
    imaging_df = imaging_df.rename(columns={ID_COL: "ID"})
    imaging_df["ID"] = imaging_df["ID"].astype(str).str.strip()

    # Date column mapping for differnt files' data names
    DATE_MAP = {
        "mri": "SCANDT",
        "fdg_pet": "SCANDATE",
        "amyloid_pet": "SCANDATE",
        "tau_pet": "SCANDATE",
    }

    DATE_COL = DATE_MAP[modality]

    imaging_df = imaging_df.rename(columns={DATE_COL: "IMAGE_DATE"})
    imaging_df["IMAGE_DATE"] = pd.to_datetime(imaging_df["IMAGE_DATE"], errors="coerce")
    
    imaging_df = imaging_df.copy()
    imaging_df["MERGE_DATE"] = imaging_df["IMAGE_DATE"]

    # print("\nImaging ID and date preview:")
    # print(imaging_df[["ID", "IMAGE_DATE"]].head(10))

    imaging_df = imaging_df.dropna(subset=["ID", "IMAGE_DATE"]).copy()
    imaging_df = imaging_df.sort_values(["ID", "IMAGE_DATE"]).reset_index(drop=True) # Sort for merge later

    # print("Prepared imaging shape:", imaging_df.shape)
    # print("Unique imaging IDs:", imaging_df["ID"].nunique())
    

    return imaging_df


def prepare_visit_df(visit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize visitation dataframe to have:
    - ID
    - VISIT_DATE
    - MERGE_DATE
    """
    print("\n=== PREPARING VISIT DATA ===")
    # print("Original visit shape:", visit_df.shape)
    print("Visit columns:")
    print(visit_df.columns.tolist())

    # Visit ID col
    ID_COL = "NACCID"
    visit_df = visit_df.rename(columns={ID_COL: "ID"})
    visit_df["ID"] = visit_df["ID"].astype(str).str.strip()
    
    # Replace missing data values with NaN
    target_cols = ["NACCMMSE", "CDRSUM", "MEMORY", "CDRLANG", "DEMENTED"]
    missing_codes = [-4, 88, 95, 96, 97, 98]

    # Replace them with NaN
    visit_df[target_cols] = visit_df[target_cols].replace(missing_codes, pd.NA)

    # Visit date split, convert to datetime
    visit_df["VISIT_DATE"] = pd.to_datetime(
        {
            "year": pd.to_numeric(visit_df["VISITYR"], errors="coerce"),
            "month": pd.to_numeric(visit_df["VISITMO"], errors="coerce"),
            "day": pd.to_numeric(visit_df["VISITDAY"], errors="coerce"),
        },
        errors="coerce"
    )

    visit_df = visit_df.copy()
    visit_df["MERGE_DATE"] = visit_df["VISIT_DATE"]

    # print("\nVisit ID/date preview:")
    # print(visit_df[["ID", "VISIT_DATE"]].head(10))

    visit_df = visit_df.dropna(subset=["ID", "VISIT_DATE"]).copy()
    visit_df = visit_df.sort_values(["ID", "VISIT_DATE"]).reset_index(drop=True)

    # print("Prepared visit shape:", visit_df.shape)
    # print("Unique visit IDs:", visit_df["ID"].nunique())
    

    return visit_df


def merge_imaging_with_visits(
    imaging_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    tolerance_days: int = 180,
) -> pd.DataFrame:
    """
    For each imaging row, attach the nearest visit row for the same ID,
    within tolerance_days.
    """
    print("\n=== MERGING IMAGING WITH VISITS ===")
    print(f"Merge mode: nearest")
    print(f"Tolerance (days): {tolerance_days}")

    # Get the needed cols lined up for merge
    imaging_for_merge = imaging_df.sort_values(["MERGE_DATE", "ID"]).reset_index(drop=True)
    visit_for_merge = visit_df.sort_values(["MERGE_DATE", "ID"]).reset_index(drop=True)

    # Merge with the closest match
    merged = pd.merge_asof(
        imaging_for_merge,
        visit_for_merge,
        by="ID",
        on="MERGE_DATE",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
        suffixes=("_img", "_visit"),
    )

    # Track the gap between exact visits
    if "IMAGE_DATE" in merged.columns and "VISIT_DATE" in merged.columns:
        merged["DATE_GAP_DAYS"] = (merged["IMAGE_DATE"] - merged["VISIT_DATE"]).abs().dt.days

    matched_mask = merged["VISIT_DATE"].notna() if "VISIT_DATE" in merged.columns else pd.Series(False, index=merged.index)
    matched_rows = int(matched_mask.sum())
    unmatched_rows = int((~matched_mask).sum())

    print("\nMerge result summary:")
    print("Merged shape:", merged.shape)
    print("Matched imaging rows:", matched_rows)
    print("Unmatched imaging rows:", unmatched_rows)

    if "DATE_GAP_DAYS" in merged.columns and matched_rows > 0:
        print("\nDate gap summary (days):")
        print(merged.loc[matched_mask, "DATE_GAP_DAYS"].describe())
        
    # Drop rows where there is no clinical data
    print(f"\nDropping {unmatched_rows} rows where there is no associated visit data.")
    merged = merged.dropna(subset=["VISIT_DATE"])

    print("\nMerged date preview:")
    preview_cols = [col for col in ["ID", "IMAGE_DATE", "VISIT_DATE", "DATE_GAP_DAYS"] if col in merged.columns]
    print(merged[preview_cols].head(15))

    return merged, matched_rows, unmatched_rows


def build_merged_dataset(
    data_dir: str,
    modality: str = "mri",
    visit_zip_name: str = "NACC_visitation_data.zip",
    visit_csv_name: str = "investigator_ftldlbd_nacc72.csv",
    tolerance_days: int = 180,
) -> pd.DataFrame:
    """
    load imaging + visits, standardize, and merge.
    """
    imaging_df = load_imaging_data(data_dir=data_dir, modality=modality)
    visit_df = load_visitation_data(
        data_dir=data_dir,
        zip_name=visit_zip_name,
        csv_name=visit_csv_name
    )

    print("\n=== DATA LOAD SUMMARY ===")
    print("Loaded imaging shape:", imaging_df.shape)
    print("Loaded visits shape:", visit_df.shape)

    # Prep for the merge
    imaging_df = prepare_imaging_df(imaging_df, modality)
    visit_df = prepare_visit_df(visit_df)

    merged_df, matched_rows, unmatched_rows = merge_imaging_with_visits(
        imaging_df=imaging_df,
        visit_df=visit_df,
        tolerance_days=tolerance_days,
    )

    print("\n=== FINAL MERGED DATASET SUMMARY ===")
    original_imaging_rows = imaging_df.shape[0]
    print("Original imaging row count: ", original_imaging_rows)
    print("Matched rows: ", matched_rows)
    print("Unmatched rows: ", unmatched_rows)
    print("Merged shape:", merged_df.shape)

    return merged_df, original_imaging_rows, matched_rows, unmatched_rows 


if __name__ == "__main__":
    data_dir = "data"
    modality = "fdg_pet"
    visit_zip_name = "NACC_visitation_data.zip"
    visit_csv_name = "investigator_ftldlbd_nacc72.csv"
    tolerance_days = 180
    
    print(f"\nConfig:")
    print(f"  data_dir = {data_dir}")
    print(f"  modality = {modality}")
    print(f"  visit_zip_name = {visit_zip_name}")
    print(f"  visit_csv_name = {visit_csv_name}")
    print(f"  tolerance_days = {tolerance_days}")
    print()
    
    build_merged_dataset(data_dir, modality, visit_zip_name, visit_csv_name, tolerance_days)


