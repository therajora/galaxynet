"""
S-PLUS Galaxy Image Downloader Module (Step 2).
Downloads galaxy images from S-PLUS using classified data from SDSS step.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from PIL import Image
import splusdata

from collect.config import (
    SPLUS_USER, SPLUS_PASS, CLASSIFIED_DATA_FILE, SPLUS_IMAGE_DIR,
    SPLUS_IMAGE_SIZE, SPLUS_IMAGE_STRETCH, LABEL_TO_FOLDER_MAP
)
from collect.utils import get_resume_index, log_download, create_pgc_id


class SPLUSDownloader:
    """Handles S-PLUS galaxy image downloads."""
    
    def __init__(self):
        self.log_file = os.path.join(SPLUS_IMAGE_DIR, "download_log.txt")
        
    def connect_to_splus(self):
        """Connects to the S-PLUS database using credentials from config."""
        print("Connecting to S-PLUS...")
        if not SPLUS_USER or not SPLUS_PASS:
            print("Error: SPLUS_USER or SPLUS_PASS not found in the .env file.")
            sys.exit(1)
        try:
            conn = splusdata.Core(SPLUS_USER, SPLUS_PASS)
            print("Connection to S-PLUS successful!")
            return conn
        except Exception as e:
            print(f"S-PLUS connection failed: {e}")
            sys.exit(1)
    
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Loads and prepares the galaxy data from a CSV file."""
        print(f"Loading data from '{file_path}'...")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: Data file '{file_path}' not found.")
            sys.exit(1)
        
        df = df.drop_duplicates(subset=["PGC"])
        df = df[df["Label"] != "Unknown"].copy()
        df["GeneralType"] = df["Label"].map(LABEL_TO_FOLDER_MAP)
        df.dropna(subset=['RA', 'DEC', 'GeneralType'], inplace=True)
        
        print(f"Data prepared. Found {len(df)} classified galaxies to process.")
        print("Class distribution:")
        print(df["GeneralType"].value_counts())
        return df
    
    def download_galaxy_image(self, galaxy_row: pd.Series, splus_conn):
        """Downloads a single galaxy image, saves it, and logs the result."""
        ra, dec, galaxy_type = galaxy_row['RA'], galaxy_row['DEC'], galaxy_row['GeneralType']
        
        pgc_id = galaxy_row.get('PGC')
        pgc_str = create_pgc_id(pgc_id)
        file_name = f"{pgc_str}.jpg"
        
        target_folder = os.path.join(SPLUS_IMAGE_DIR, galaxy_type)
        file_path = os.path.join(target_folder, file_name)
        os.makedirs(target_folder, exist_ok=True)
        
        if os.path.exists(file_path):
            return  # Skip if already downloaded
        
        try:
            splus_image = splus_conn.lupton_rgb(ra, dec, size=SPLUS_IMAGE_SIZE, stretch=SPLUS_IMAGE_STRETCH)
            if isinstance(splus_image, Image.Image):
                splus_image.save(file_path, "JPEG")
                log_status = "Success"
            else:
                log_status = "Error: Empty image returned"
        except Exception as e:
            log_status = f"Error: {str(e).replace(',', ';')}"  # Avoid CSV conflicts
        
        # Log the outcome
        log_download(self.log_file, pgc_str, log_status)
    
    def download_images(self, df: pd.DataFrame, splus_conn):
        """Downloads images for all galaxies in the DataFrame."""
        start_index = get_resume_index(df, self.log_file)
        df_to_process = df.iloc[start_index:]
        
        if df_to_process.empty:
            print("\nAll galaxies have already been processed according to the log file.")
            return
        
        if start_index > 0:
            print(f"\n>>> Resuming download after index {start_index}. <<<")
        
        print(f"\nStarting download for {len(df_to_process)} galaxies...")
        
        # Use tqdm to create a progress bar
        for _, galaxy in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Downloading S-PLUS Images"):
            self.download_galaxy_image(galaxy, splus_conn)
        
        print("\nS-PLUS image download process completed!")
    
    def run(self):
        """Main execution method."""
        splus_connection = self.connect_to_splus()
        df = self.load_and_prepare_data(CLASSIFIED_DATA_FILE)
        self.download_images(df, splus_connection)


def main():
    """Main function for standalone execution."""
    downloader = SPLUSDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
