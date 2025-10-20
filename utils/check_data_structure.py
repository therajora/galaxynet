"""
Script to verify organized data structure.
"""

from pathlib import Path


def check_data_structure():
    """Verify if the data structure is correct."""
    print("=== Checking Data Structure ===\n")
    
    required_paths = [
        "data/csv/rc3_raw.csv",
        "data/csv/rc3_classified.csv", 
        "data/images/sdss/reg",
        "data/images/sdss/irr_pec",
        "data/images/splus/reg",
        "data/images/splus/irr_pec",
        "data/logs"
    ]
    
    all_good = True
    for path in required_paths:
        if Path(path).exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} - NOT FOUND")
            all_good = False
    
    if all_good:
        print("\n✅ Data structure is correct!")
    else:
        print("\n❌ Incomplete data structure.")
        return False
    
    return True


def count_images():
    """Count the number of images in each folder."""
    print("\n=== Image Count ===\n")
    
    image_dirs = [
        ("SDSS Regular", "data/images/sdss/reg"),
        ("SDSS Irregular/Peculiar", "data/images/sdss/irr_pec"),
        ("S-PLUS Regular", "data/images/splus/reg"),
        ("S-PLUS Irregular/Peculiar", "data/images/splus/irr_pec")
    ]
    
    total_images = 0
    for name, path in image_dirs:
        if Path(path).exists():
            count = len([f for f in Path(path).iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"{name}: {count} images")
            total_images += count
        else:
            print(f"{name}: Folder not found")
    
    print(f"\nTotal images: {total_images}")


def show_structure():
    """Show the created data structure."""
    print("\n=== Created Data Structure ===\n")
    print("data/")
    print("├── csv/")
    print("│   ├── rc3_raw.csv")
    print("│   └── rc3_classified.csv")
    print("├── images/")
    print("│   ├── sdss/")
    print("│   │   ├── reg/")
    print("│   │   └── irr_pec/")
    print("│   └── splus/")
    print("│       ├── reg/")
    print("│       └── irr_pec/")
    print("└── logs/")
    print("    ├── sdss_download_log.txt")
    print("    └── splus_download_log.txt")


def main():
    """Main function."""
    print("=== Data Structure Verification ===\n")
    
    # Check data structure
    if check_data_structure():
        # Count images
        count_images()
        
        # Show structure
        show_structure()
        
        print("\n=== Next Steps ===\n")
        print("1. Install dependencies:")
        print("   ./install_dependencies.sh")
        print()
        print("2. Configure your credentials in the .env file:")
        print("   cp env.example .env")
        print("   # Edit the .env file with your credentials")
        print()
        print("3. Use the collection modules:")
        print("   from collect import SDSSDownloader, SPLUSDownloader")


if __name__ == "__main__":
    main()