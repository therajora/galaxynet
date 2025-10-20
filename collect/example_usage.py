"""
Example usage of the galaxy data collection modules.

This script demonstrates how to use both the SDSS and S-PLUS downloaders
in sequence to collect galaxy images for classification.
"""

from collect import SDSSDownloader, SPLUSDownloader


def main():
    """Run the complete data collection pipeline."""
    print("=== Galaxy Data Collection Pipeline ===\n")
    
    # Step 1: Download and classify SDSS images
    print("Step 1: Downloading and classifying SDSS images...")
    sdss_downloader = SDSSDownloader()
    sdss_downloader.run()
    
    print("\n" + "="*50 + "\n")
    
    # Step 2: Download S-PLUS images using classified data
    print("Step 2: Downloading S-PLUS images...")
    splus_downloader = SPLUSDownloader()
    splus_downloader.run()
    
    print("\n=== Data Collection Pipeline Completed! ===")


if __name__ == "__main__":
    main()
