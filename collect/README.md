# Galaxy Data Collection Module

Simple module for downloading galaxy images from SDSS and S-PLUS surveys.

## Files

- `sdss_downloader.py` - Downloads images from SDSS DR18 and classifies galaxies
- `splus_downloader.py` - Downloads images from S-PLUS for classified galaxies
- `utils.py` - Helper functions for classification and logging
- `config.py` - Configuration settings and credentials
- `example_usage.py` - Usage examples

## Quick Usage

```python
from collect import SDSSDownloader, SPLUSDownloader

# Step 1: Download and classify SDSS images
sdss = SDSSDownloader()
sdss.run()

# Step 2: Download S-PLUS images
splus = SPLUSDownloader()
splus.run()
```

## What It Does

1. Reads RC3 catalog data with galaxy coordinates
2. Downloads images from SDSS using SkyServer API
3. Classifies galaxies as Regular or Peculiar based on T-Type and Hubble code
4. Downloads matching images from S-PLUS using coordinates
5. Organizes images into class folders (reg, irr_pec)

## Classification Logic

- **T-Type**: Primary classification criterion (-4 to 9 = Regular, >=10 = Peculiar)
- **Hubble Code**: Fallback and refinement (checks for .P or ? markers)
- **Final Classes**: Regular (reg) and Irregular/Peculiar (irr_pec)

## Requirements

- SciServer credentials for SDSS access
- S-PLUS credentials for S-PLUS access
- Store credentials in `.env` file

