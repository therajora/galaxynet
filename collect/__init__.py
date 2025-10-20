"""
Galaxy Data Collection Module.

This module provides functionality for downloading and processing galaxy images
from both SDSS and S-PLUS surveys. It consists of two main steps:

1. SDSS Downloader: Downloads galaxy images from SDSS and classifies them
2. S-PLUS Downloader: Downloads corresponding images from S-PLUS using classified data

Usage:
    from collect import SDSSDownloader, SPLUSDownloader
    
    # Step 1: Download and classify SDSS images
    sdss_downloader = SDSSDownloader()
    sdss_downloader.run()
    
    # Step 2: Download S-PLUS images using classified data
    splus_downloader = SPLUSDownloader()
    splus_downloader.run()
"""

from .sdss_downloader import SDSSDownloader
from .splus_downloader import SPLUSDownloader
from .utils import (
    get_morphological_class_from_T,
    get_morphological_class_from_hubble,
    get_downloaded_ids,
    log_download,
    get_resume_index,
    create_pgc_id
)

__version__ = "1.0.0"
__author__ = "Galaxy Classification Team"

__all__ = [
    'SDSSDownloader',
    'SPLUSDownloader',
    'get_morphological_class_from_T',
    'get_morphological_class_from_hubble',
    'get_downloaded_ids',
    'log_download',
    'get_resume_index',
    'create_pgc_id'
]
