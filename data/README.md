# Data Directory

Directory structure for galaxy image datasets and metadata.

## Structure

```
data/
├── csv/                    - Galaxy catalog data
│   ├── rc3_raw.csv        - Original RC3 catalog
│   └── rc3_classified.csv - Classified galaxy data
│
├── images/                 - Original downloaded images
│   ├── sdss/              - SDSS images (by class)
│   └── splus/             - S-PLUS images (by class)
│
├── complete_sdss/          - SDSS augmented and balanced dataset
│   ├── reg/               - Regular galaxies
│   └── irr_pec/           - Peculiar galaxies
│
├── complete_splus/         - S-PLUS augmented and balanced dataset
│   ├── reg/               - Regular galaxies
│   └── irr_pec/           - Peculiar galaxies
│
├── visualization/          - Scripts for creating image mosaics
├── logs/                   - Download and processing logs
└── mosaics/                - Generated mosaic visualizations
```

## Files

- `data_summary.txt` - Summary of original dataset
- `data_summary_augmentation.txt` - Summary after augmentation and balancing

## Folders

- **csv**: Catalog files (RC3 data)
- **images**: Original downloaded images from surveys
- **complete_sdss**: Final SDSS dataset (augmented + balanced)
- **complete_splus**: Final S-PLUS dataset (augmented + balanced)
- **visualization**: Tools to create image mosaics
- **mosaics**: Output mosaic images
- **logs**: Processing logs

## Dataset Flow

1. Raw catalog → `csv/rc3_raw.csv`
2. Classification → `csv/rc3_classified.csv`
3. Download → `images/sdss/` and `images/splus/`
4. Augmentation + balancing → `complete_sdss/` and `complete_splus/`

