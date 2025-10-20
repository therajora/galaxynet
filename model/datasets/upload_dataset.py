#!/usr/bin/env python3
"""
CLI script for uploading datasets to Hugging Face Hub.

This script allows creating and uploading galaxy datasets to Hugging Face Hub
in a simple and direct way.
"""

import argparse
import os
import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if they exist
                    value = value.strip('"\'')
                    os.environ[key] = value

# Load .env file
load_env_file()

from model.datasets.galaxy_dataset import (
    GalaxyDataset, create_huggingface_dataset, upload_to_hub,
    create_binary_dataset, split_and_upload_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Upload galaxy datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

  # Basic upload
  python upload_dataset.py -d data/complete_sdss -r "username/galaxy-dataset"

  # Upload with train/test split
  python upload_dataset.py -d data/complete_sdss -r "username/galaxy-dataset" --split 0.2

  # Upload binary dataset
  python upload_dataset.py -d data/complete_sdss -r "username/galaxy-binary" --binary

  # Private upload with specific token
  python upload_dataset.py -d data/complete_sdss -r "username/galaxy-dataset" --private --token "hf_xxx"

  # Only create local dataset (no upload)
  python upload_dataset.py -d data/complete_sdss --create-only
        """
    )

    # Required arguments
    parser.add_argument(
        "-d", "--data-dir",
        required=True,
        help="Directory with images organized by class"
    )

    # Optional arguments
    parser.add_argument(
        "-r", "--repo-id",
        help="Repository ID on Hugging Face (e.g., username/dataset-name)"
    )

    parser.add_argument(
        "--split",
        type=float,
        default=0.0,
        help="Proportion for train/test split (0.0 = no split)"
    )

    parser.add_argument(
        "--binary",
        action="store_true",
        help="Create binary dataset (Regular vs Peculiar)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as private repository"
    )

    parser.add_argument(
        "--token",
        help="Hugging Face token (optional, uses HUGGINGFACE_TOKEN if not specified)"
    )

    parser.add_argument(
        "--commit-message",
        default="Upload galaxy dataset",
        help="Commit message"
    )

    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create local dataset, no upload"
    )

    parser.add_argument(
        "--metadata-path",
        help="Path to save metadata (default: data_dir/galaxy_metadata.pth)"
    )

    args = parser.parse_args()

    # Validations
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found")
        sys.exit(1)

    if not args.create_only and not args.repo_id:
        print("Error: --repo-id is required when not using --create-only")
        sys.exit(1)

    if args.split < 0 or args.split >= 1:
        print("Error: --split must be between 0 and 1")
        sys.exit(1)

    # Check token if necessary
    if not args.create_only:
        if not args.token and not os.getenv('HUGGINGFACE_TOKEN'):
            print("Error: Hugging Face token not found.")
            print("Set HUGGINGFACE_TOKEN or use --token")
            sys.exit(1)

    print("="*60)
    print("UPLOAD DATASET TO HUGGING FACE HUB")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    if args.repo_id:
        print(f"Repository: {args.repo_id}")
    print(f"Binary dataset: {args.binary}")
    print(f"Train/test split: {args.split}")
    print(f"Private repository: {args.private}")
    print(f"Create local only: {args.create_only}")
    print("="*60)

    try:
        if args.create_only:
            # Only create local dataset
            print("\nCreating local dataset...")
            dataset = GalaxyDataset(
                img_dir=args.data_dir,
                metadata_path=args.metadata_path
            )
            dataset.print_summary()
            print("Local dataset created successfully!")
            
        elif args.split > 0:
            # Upload with split
            print(f"\nCreating dataset and splitting into train/test...")
            success = split_and_upload_dataset(
                img_dir=args.data_dir,
                repo_id=args.repo_id,
                test_size=args.split,
                binary=args.binary,
                commit_message=args.commit_message,
                token=args.token,
                private=args.private
            )
            
            if success:
                print("Upload with split completed successfully!")
            else:
                print("Error during upload with split")
                sys.exit(1)
                
        else:
            # Simple upload
            print(f"\nCreating dataset...")
            if args.binary:
                dataset = create_binary_dataset(args.data_dir)
            else:
                dataset = create_huggingface_dataset(args.data_dir)
            
            print(f"Dataset created with {len(dataset)} samples")
            
            print(f"\nUploading...")
            success = upload_to_hub(
                dataset=dataset,
                repo_id=args.repo_id,
                commit_message=args.commit_message,
                token=args.token,
                private=args.private
            )
            
            if success:
                print("Upload completed successfully!")
            else:
                print("Error during upload")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    print("\nOperation completed!")


if __name__ == "__main__":
    main()
