"""
Configuration module for galaxy data collection.
Contains shared settings and mappings for both SDSS and S-PLUS downloaders.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# AUTHENTICATION CREDENTIALS
# =============================================================================
SCISERVER_USER = os.getenv('SCISERVER_USER')
SCISERVER_PASS = os.getenv('SCISERVER_PASS')
SPLUS_USER = os.getenv('SPLUS_USER')
SPLUS_PASS = os.getenv('SPLUS_PASS')

# =============================================================================
# FILE PATHS
# =============================================================================
RAW_DATA_FILE = "data/csv/rc3_raw.csv"
CLASSIFIED_DATA_FILE = "data/csv/rc3_classified.csv"
SDSS_IMAGE_DIR = "data/images/sdss"
SPLUS_IMAGE_DIR = "data/images/splus"

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
# SDSS Settings
SDSS_IMG_SCALE = 0.6
SDSS_IMG_HEIGHT = 224
SDSS_IMG_WIDTH = 224

# S-PLUS Settings
SPLUS_IMAGE_SIZE = 240
SPLUS_IMAGE_STRETCH = 5

# =============================================================================
# CLASSIFICATION MAPPINGS
# =============================================================================
# Maps detailed classes to broader labels for folder organization
CLASS_TO_LABEL_MAP = {
    "Spirals": "reg",
    "Lenticulars": "reg",
    "Ellipticals": "reg",
    "Spiral (Peculiar)": "irr_pec",
    "Irregulars": "irr_pec",
    "Lenticular (Peculiar)": "irr_pec",
    "Elliptical (Peculiar)": "irr_pec",
    "Irregular (Peculiar)": "irr_pec",
    "Peculiar": "irr_pec",
    "Unknown": "Unknown"
}

# Maps labels from the input CSV to folder names (for S-PLUS)
LABEL_TO_FOLDER_MAP = {
    "Irregular/Peculiar": "irr_pec",
    "Regular": "reg"
}
