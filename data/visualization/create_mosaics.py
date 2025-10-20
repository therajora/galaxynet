"""
Script to create image mosaics.

This script creates class-organized mosaics for visualizing galaxy data.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.visualization.mosaic_generator import MosaicGenerator, create_mosaics_for_project


def main():
    parser = argparse.ArgumentParser(description="Create galaxy image mosaics")
    
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "single", "comparison"],
                       help="Operation mode: all (all), single (one dataset), comparison (comparative)")
    
    parser.add_argument("--dataset", type=str,
                       help="Dataset name (e.g., sdss, splus)")
    
    parser.add_argument("--data-dir", type=str,
                       help="Directory with class-organized data")
    
    parser.add_argument("--output-dir", type=str, default="data/mosaics",
                       help="Directory to save mosaics")
    
    parser.add_argument("--class-names", nargs=2, default=None,
                       help="Class names (if not specified, auto-detects)")
    
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum images per class")
    
    args = parser.parse_args()
    
    # Create generator
    generator = MosaicGenerator(output_dir=Path(args.output_dir))
    
    if args.mode == "all":
        print("Creating all mosaics...")
        mosaics = create_mosaics_for_project()
        
    elif args.mode == "single":
        if not args.dataset and not args.data_dir:
            parser.error("For 'single' mode, specify --dataset or --data-dir")
        
        if args.dataset:
            data_dir = Path(f"data/images/{args.dataset}")
            mosaic_name = f"{args.dataset}_mosaic"
        else:
            data_dir = Path(args.data_dir)
            mosaic_name = data_dir.name + "_mosaic"
        
        if not data_dir.exists():
            print(f"Error: Directory not found: {data_dir}")
            sys.exit(1)
        
        print(f"Creating mosaic for: {data_dir}")
        mosaic_path = generator.create_mosaic(
            data_dir=data_dir,
            class_names=args.class_names,
            mosaic_name=mosaic_name,
            max_images_per_class=args.max_images
        )
        mosaics = [mosaic_path]
        
    elif args.mode == "comparison":
        sdss_dir = Path("data/images/sdss")
        splus_dir = Path("data/images/splus")
        
        if not sdss_dir.exists():
            print(f"Error: SDSS directory not found: {sdss_dir}")
            sys.exit(1)
        
        if not splus_dir.exists():
            print(f"Error: S-PLUS directory not found: {splus_dir}")
            sys.exit(1)
        
        print("Creating comparison mosaic...")
        mosaic_path = generator.create_comparison_mosaic(
            sdss_dir=sdss_dir,
            splus_dir=splus_dir,
            class_names=args.class_names
        )
        mosaics = [mosaic_path]
    
    print(f"\nSuccess: {len(mosaics)} mosaic(s) created:")
    for mosaic in mosaics:
        print(f"  - {mosaic}")


if __name__ == "__main__":
    main()
