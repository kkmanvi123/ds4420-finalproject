import os
import zipfile
import pandas as pd

def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract if target folder doesnt exist

    Args:
        zip_path (str): zip file path
        extract_dir (str): dir to extract into
    """
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
        "mri": os.path.join(mri_extract_dir, "investigator_scan_mrisbm_nacc72.csv"),
        "fdg_pet": os.path.join(pet_extract_dir, "investigator_scan_fdgpetnpdka_nacc72.csv"),
        "amyloid_pet": os.path.join(pet_extract_dir, "investigator_scan_amyloidpetnpdka_nacc72.csv"),
        "tau_pet": os.path.join(pet_extract_dir, "investigator_scan_taupetnpdka_nacc72.csv"),
    }

    data = {}

    # create the dict
    for key, file_path in file_map.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        data[key] = pd.read_csv(file_path)

    return data



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





