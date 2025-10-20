"""
Utility functions for galaxy data collection.
Contains shared helper functions for both SDSS and S-PLUS downloaders.
"""

import os
import pandas as pd
from typing import Set


def get_morphological_class_from_T(T_value: float) -> str:
    """Classifies a galaxy based on its numerical T-Type value."""
    if pd.isna(T_value) or abs(int(T_value)) == 99:
        return "Unknown"
    t_int = round(T_value)
    if t_int <= -4:
        return "Ellipticals"
    if -3 <= t_int <= -1:
        return "Lenticulars"
    if 0 <= t_int <= 9:
        return "Spirals"
    if t_int in (10, 90, 11):
        return "Irregulars"
    return "Unknown"


def get_morphological_class_from_hubble(code: str) -> str:
    """Classifies a galaxy based on its Hubble string code as a fallback."""
    if not isinstance(code, str) or not code:
        return "Unknown"
    c = code.upper().strip()
    if ".P" in c or "?" in c:
        if ".S" in c:
            return "Spiral (Peculiar)"
        if ".E" in c or c.startswith("DE"):
            return "Elliptical (Peculiar)"
        if ".L" in c:
            return "Lenticular (Peculiar)"
        if ".I" in c:
            return "Irregular (Peculiar)"
        return "Peculiar"
    if ".S" in c:
        return "Spirals"
    if ".L" in c:
        return "Lenticulars"
    if ".E" in c or c.startswith("DE"):
        return "Ellipticals"
    if ".I" in c or ".RING" in c or c.startswith("P.A"):
        return "Irregulars"
    return "Unknown"


def get_downloaded_ids(log_file: str) -> Set[str]:
    """Reads the log file to return a set of already downloaded galaxy IDs."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r', encoding='utf-8') as f:
        downloaded_ids = {line.strip() for line in f if line.strip()}
    return downloaded_ids


def log_download(log_file: str, unique_id: str, status: str = "Success"):
    """Adds a download record to the log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{unique_id}, {status}\n")


def get_resume_index(df: pd.DataFrame, log_file: str) -> int:
    """Determines the starting index by reading the last entry in the log file."""
    if not os.path.exists(log_file):
        return 0

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return 0

        last_pgc_str = lines[-1].split(",")[0].strip()
        last_pgc = last_pgc_str.replace("PGC_", "").strip()

        # Find the index of the last processed PGC in the DataFrame
        matches = df.index[df['PGC'].astype(str).str.strip() == last_pgc].tolist()
        if matches:
            return matches[0] + 1
        else:
            return 0
    except (ValueError, IndexError):
        return 0


def create_pgc_id(pgc_value) -> str:
    """Creates a standardized PGC ID string."""
    if pd.isna(pgc_value):
        return "Unknown"
    return f"PGC_{str(pgc_value).replace('PGC', '').strip()}"
