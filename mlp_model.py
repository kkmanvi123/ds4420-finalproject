import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract if target folder doesnt exist

    Args:
        zip_path (str): zip file path
        extract_dir (str): dir to extract into
    """
    print()
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

def pull_data(data_dir: str) -> dict:
    """Load the appropriate MRI/PET files for PCA -> MLP model

    Args:
        data_dir (str): path of the data directory

    Returns:
        dict: dictionary of dataframes
    """
    
    # unzip if needed
    mri_zip = os.path.join(data_dir, "NACC_mri_data.zip")
    pet_zip = os.path.join(data_dir, "NACC_pet_data.zip")

    mri_extract_dir = os.path.join(data_dir, "NACC_mri_data")
    pet_extract_dir = os.path.join(data_dir, "NACC_pet_data")

    extract_zip(mri_zip, mri_extract_dir)
    extract_zip(pet_zip, pet_extract_dir)
    
    # pull the files
    file_map = {
        "mri": os.path.join(mri_extract_dir, "investigator_scan_mrisbm_nacc72.csv"), # structure of the brain (shrinkage)
        "fdg_pet": os.path.join(pet_extract_dir, "investigator_scan_fdgpetnpdka_nacc72.csv"), # neuron activity, detect before shrinkage
        "amyloid_pet": os.path.join(pet_extract_dir, "investigator_scan_amyloidpetnpdka_nacc72.csv"), # Amyloid plaque buildup
        "tau_pet": os.path.join(pet_extract_dir, "investigator_scan_taupetnpdka_nacc72.csv"), # tau protein tangles in brain (cognitive decline)
    }

    data = {}

    # create the dict
    for key, file_path in file_map.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        data[key] = pd.read_csv(file_path)

    return data

def clean_feaures(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe to only have feature columns for PCA

    Args:
        df (pd.DataFrame): in df

    Returns:
        pd.DataFrame: cleaned df
    """
    
    # common id cols to drop (intake info)
    drop_cols = [
        "NACCID", "NACCADC", "LONIUID", "LONIUID_MULTI"
        "SCANDATE", "PROCESSDATE",
        "TRACER", "ACQUISITION_TIME"
    ]
    
    # drop known non-feature columns
    df_clean = df.drop(columns=drop_cols, errors="ignore")

    # remove any non-numeric other data
    df_clean = df_clean.select_dtypes(include=["number"])

    return df_clean

def run_pca(df: pd.DataFrame, n: float = 30) -> np.ndarray:    
    """run PCA algorithm on given data

    Args:
        df (pd.DataFrame): input datafram
        n (float, optional): number of components OR variance%. Defaults to 30.

    Returns:
        np.ndarray: reduced components as an array
    """

    X = df.dropna() # simple na drop can fix later
    
    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # perform PCA
    pca = PCA(n_components=n)  # number of comonents or variance%
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA with {n}% variance" if n < 1 else f"PCA with {n} comonents")
    print("Original shape:", X.shape)
    print("PCA shape:", X_pca.shape)

    return X_pca


def main():
    data = pull_data("data")

    for name, df in data.items():
        print(f"\n=== {name.upper()} ===")

        # shape
        print(f"Shape: {df.shape}")

        # first few rows
        print("\nHead:")
        print(df.head())


if __name__ == "__main__":
    main()





