#!/usr/bin/env python3

import gdown
import zipfile
import io
from pathlib import Path
import argparse

FILE_ID = "15zLaP5V3ltR6qro98u492eqFBSeSO-0m"

def is_relative_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Only relative paths are allowed.")
    return path

def main():
    parser = argparse.ArgumentParser(
        description="Download a ZIP dataset from Google Drive (in memory) and extract it to a relative folder."
    )
    parser.add_argument(
        "--extract_to",
        type=is_relative_path,
        default="dataset",
        help="Relative folder to extract the dataset to (default: 'dataset')"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    extract_dir = script_dir / args.extract_to
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Build full download URL for gdown
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    try:
        print(f"Downloading ZIP file from Google Drive (file ID: {FILE_ID})...")
        zip_data = gdown.download(url, quiet=False, fuzzy=True, use_cookies=False, resume=False, output=None)

        if zip_data is None:
            raise RuntimeError("gdown failed to download the file.")

        # Open the ZIP file as bytes in memory
        with open(zip_data, 'rb') as f:
            zip_bytes = io.BytesIO(f.read())

        with zipfile.ZipFile(zip_bytes) as zip_file:
            print(f"ZIP contains: {zip_file.namelist()}")
            zip_file.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")

        # Delete the temporary ZIP file downloaded by gdown
        Path(zip_data).unlink()
        print(f"Temporary ZIP file deleted: {zip_data}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
