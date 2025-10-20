"""
SDSS Galaxy Image Downloader Module (Step 1).
Downloads galaxy images from SDSS and classifies them based on morphological data.
"""

import os
import sys
import requests
import pandas as pd
from tqdm import tqdm
from SciServer import Authentication

from collect.config import (
    SCISERVER_USER, SCISERVER_PASS, RAW_DATA_FILE, CLASSIFIED_DATA_FILE,
    SDSS_IMAGE_DIR, SDSS_IMG_SCALE, SDSS_IMG_HEIGHT, SDSS_IMG_WIDTH,
    CLASS_TO_LABEL_MAP
)
from collect.utils import (
    get_morphological_class_from_T, get_morphological_class_from_hubble,
    get_downloaded_ids, log_download, create_pgc_id
)


class SDSSDownloader:
    """Handles SDSS galaxy image downloads and classification."""
    
    def __init__(self):
        self.log_file = os.path.join(SDSS_IMAGE_DIR, 'download_log_sdss.txt')
        
    def authenticate(self):
        """Authenticates with SciServer using credentials from config."""
        print("Authenticating with SciServer...")
        if not SCISERVER_USER or not SCISERVER_PASS:
            print("Error: SCISERVER_USER or SCISERVER_PASS not found in .env file.")
            sys.exit(1)
        try:
            Authentication.login(SCISERVER_USER, SCISERVER_PASS)
            print("SciServer authentication successful!")
        except Exception as e:
            print(f"Authentication failed: {e}")
            sys.exit(1)
    
    def classify_galaxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies classification logic to the DataFrame."""
        print("Classifying galaxies...")
        df = df.copy()
        df["Class_TType"] = df["TYPE"].apply(get_morphological_class_from_T)
        df["Class_Hubble"] = df["HUBBLE"].apply(get_morphological_class_from_hubble)
        
        # Prioritize T-Type classification, using Hubble as fallback
        df['Final_Class'] = df['Class_TType']
        df.loc[df['Final_Class'] == 'Unknown', 'Final_Class'] = df['Class_Hubble']
        
        # Map detailed classes to broader labels
        df["Label"] = df["Final_Class"].map(CLASS_TO_LABEL_MAP)
        df['Image_Folder'] = df['Label']
        
        # Clean up and remove duplicates
        df.drop_duplicates(subset=["PGC"], inplace=True)
        return df
    
    def download_image(self, url: str, file_path: str) -> bool:
        """Downloads a single image from URL to specified path."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
            return False
        except requests.exceptions.RequestException:
            return False
    
    def download_galaxy_image(self, row: pd.Series, downloaded_ids: set):
        """Processes a DataFrame row to download the corresponding galaxy image."""
        pgc_id = row.get('PGC')
        unique_id = create_pgc_id(pgc_id)
        
        if unique_id in downloaded_ids:
            return
            
        image_url = (f"http://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?"
                    f"ra={row['RA']}&dec={row['DEC']}&scale={SDSS_IMG_SCALE}"
                    f"&height={SDSS_IMG_HEIGHT}&width={SDSS_IMG_WIDTH}")
        
        folder = row['Image_Folder']
        file_path = os.path.join(SDSS_IMAGE_DIR, folder, f"{unique_id}.jpg")
        
        if self.download_image(image_url, file_path):
            log_download(self.log_file, unique_id)
    
    def process_data(self) -> pd.DataFrame:
        """Loads, classifies and saves the galaxy data."""
        try:
            print(f"Loading data from '{RAW_DATA_FILE}'...")
            df_raw = pd.read_csv(RAW_DATA_FILE)
            print(f"Successfully loaded. Shape: {df_raw.shape}")
        except FileNotFoundError:
            print(f"Error: Raw data file '{RAW_DATA_FILE}' not found.")
            sys.exit(1)
        
        df_processed = self.classify_galaxies(df_raw)
        
        print("\nFinal class distribution:")
        print(df_processed["Final_Class"].value_counts())
        
        df_processed.to_csv(CLASSIFIED_DATA_FILE, index=False)
        print(f"Classified data saved to '{CLASSIFIED_DATA_FILE}'.")
        
        return df_processed
    
    def download_images(self, df: pd.DataFrame):
        """Downloads images for classified galaxies."""
        df_to_download = df[df['Label'] != 'Unknown'].copy()
        if df_to_download.empty:
            print("No classified galaxies found to download.")
            return
        
        print(f"Preparing to download images for {len(df_to_download)} classified galaxies...")
        os.makedirs(SDSS_IMAGE_DIR, exist_ok=True)
        downloaded_ids = get_downloaded_ids(self.log_file)
        
        print(f"Found {len(downloaded_ids)} already downloaded images. Skipping...")
        
        for _, row in tqdm(df_to_download.iterrows(), total=len(df_to_download), desc="Downloading SDSS Images"):
            self.download_galaxy_image(row, downloaded_ids)
        
        print("SDSS image download process completed!")
    
    def run(self):
        """Main execution method."""
        self.authenticate()
        df = self.process_data()
        self.download_images(df)


def main():
    """Main function for standalone execution."""
    downloader = SDSSDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
