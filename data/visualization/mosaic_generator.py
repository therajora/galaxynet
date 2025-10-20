"""
Image Mosaic Generator.

This class creates class-organized mosaics for visualizing
SDSS and S-PLUS galaxy data.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


class MosaicGenerator:
    """
    Generator for class-organized image mosaics.
    
    Creates diagrams with 4 rows:
    - Rows 1-2: First class (Regular)
    - Rows 3-4: Second class (Peculiar)
    - 5 columns of images
    - File names included in image
    """
    
    def __init__(self, output_dir: Path = Path("data/mosaics")):
        """
        Initialize mosaic generator.
        
        Args:
            output_dir: Directory to save mosaics
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mosaic settings
        self.rows = 4  # 4 rows
        self.cols = 5  # 5 columns
        self.image_size = (128, 128)  # Image size in mosaic
        self.mosaic_size = (800, 640)  # Total mosaic size
        
        # Font settings
        self.font_size = 10
        self.font_color = 'white'
        
        print(f"MosaicGenerator initialized. Output: {self.output_dir}")
    
    def create_mosaic(
        self,
        data_dir: Path,
        class_names: List[str] = None,
        mosaic_name: str = None,
        max_images_per_class: int = 10
    ) -> Path:
        """
        Create a class-organized image mosaic.
        
        Args:
            data_dir: Directory with class-organized data
            class_names: Class names (default: ['Regular', 'Peculiar'])
            mosaic_name: Output file name
            max_images_per_class: Maximum images per class
            
        Returns:
            Path to saved mosaic
        """
        if class_names is None:
            class_names = self._detect_class_names(data_dir)
        
        if mosaic_name is None:
            mosaic_name = f"mosaic_{data_dir.name}"
        
        print(f"Creating mosaic for: {data_dir}")
        print(f"Classes: {class_names}")
        
        # Load images by class
        images_by_class = self._load_images_by_class(data_dir, class_names, max_images_per_class)
        
        # Create mosaic
        mosaic_path = self._generate_mosaic(images_by_class, class_names, mosaic_name)
        
        print(f"Mosaic saved to: {mosaic_path}")
        return mosaic_path
    
    def _load_images_by_class(
        self,
        data_dir: Path,
        class_names: List[str],
        max_images: int
    ) -> Dict[str, List[Tuple[Image.Image, str]]]:
        """
        Load images organized by class.
        
        Args:
            data_dir: Directory with data
            class_names: Class names
            max_images: Maximum images per class
            
        Returns:
            Dictionary with images per class
        """
        images_by_class = {}
        
        for class_name in class_names:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory '{class_name}' not found: {class_dir}")
                images_by_class[class_name] = []
                continue
            
            # List image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(class_dir.glob(ext))
            
            # Filter only images with "aug" in name (for data augmentation mosaics)
            if "augmented" in str(data_dir) or "complete" in str(data_dir):
                image_files = [f for f in image_files if "aug" in f.name.lower()]
            
            # Select random images
            selected_files = random.sample(image_files, min(len(image_files), max_images))
            
            # Load images
            class_images = []
            for img_file in selected_files:
                try:
                    img = Image.open(img_file)
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    class_images.append((img, img_file.name))
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")
                    continue
            
            images_by_class[class_name] = class_images
            print(f"Class '{class_name}': {len(class_images)} images loaded")
        
        return images_by_class
    
    def _generate_mosaic(
        self,
        images_by_class: Dict[str, List[Tuple[Image.Image, str]]],
        class_names: List[str],
        mosaic_name: str
    ) -> Path:
        """
        Generate mosaic using matplotlib.
        
        Args:
            images_by_class: Images organized by class
            class_names: Class names
            mosaic_name: File name
            
        Returns:
            Path to saved mosaic
        """
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(self.rows, self.cols, figure=fig, hspace=0.1, wspace=0.1)
        
        # Mosaic title (commented)
        # fig.suptitle(f'Galaxy Mosaic - {mosaic_name.replace("_", " ").title()}',
        #             fontsize=16, fontweight='bold', y=0.95)
        
        # Distribute images across rows
        row_idx = 0
        
        for class_idx, class_name in enumerate(class_names):
            class_images = images_by_class[class_name]
            
            # Determine how many rows this class will occupy
            if len(class_names) == 2:
                rows_per_class = 2  # 2 rows per class
            else:
                rows_per_class = self.rows // len(class_names)
            
            # Fill class rows
            for local_row in range(rows_per_class):
                if row_idx >= self.rows:
                    break
                
                for col in range(self.cols):
                    ax = fig.add_subplot(gs[row_idx, col])
                    
                    # Calculate image index
                    img_idx = local_row * self.cols + col
                    
                    if img_idx < len(class_images):
                        img, filename = class_images[img_idx]
                        
                        # Resize image
                        img_resized = img.resize(self.image_size, Image.Resampling.LANCZOS)
                        
                        # Show image
                        ax.imshow(img_resized)
                        
                        # Add galaxy name
                        if "augmented" in mosaic_name or "complete" in mosaic_name:
                            # For data augmentation mosaics: simple name at bottom right
                            simple_name = filename.split('_aug_')[0] if '_aug_' in filename else filename.split('_')[0] + '_' + filename.split('_')[1]
                            clean_name = simple_name.replace('_', '')
                            ax.text(0.95, 0.05, clean_name,
                                   transform=ax.transAxes,
                                   fontsize=8,
                                   color='white',
                                   ha='right',
                                   va='bottom')
                        else:
                            # For original mosaics: clean name at bottom right
                            clean_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', '')
                            ax.text(0.95, 0.05, clean_name,
                                   transform=ax.transAxes,
                                   fontsize=8,
                                   color='white',
                                   ha='right',
                                   va='bottom')
                        
                        # Add class label on first image of row
                        if col == 0:
                            ax.text(-0.1, 0.5, class_name,
                                   transform=ax.transAxes,
                                   fontsize=12,
                                   fontweight='bold',
                                   color='black',
                                   verticalalignment='center',
                                   horizontalalignment='right',
                                   rotation=90)
                    else:
                        # Empty image if no more images available
                        ax.imshow(np.ones((*self.image_size, 3)) * 0.9, cmap='gray')
                    
                    # Remove axes
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                
                row_idx += 1
        
        # Save mosaic
        mosaic_path = self.output_dir / f"{mosaic_name}.png"
        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.02, wspace=0.0001, hspace=0.0001)
        plt.savefig(mosaic_path, dpi=300,
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        plt.close()
        
        return mosaic_path
    
    def _extract_aug_number(self, filename: str) -> int:
        """
        Extract augmentation number from filename.
        Ex: PGC_10006_aug_sdss_1.jpg -> 1
        
        Args:
            filename: File name
            
        Returns:
            Augmentation number or 0 if not found
        """
        try:
            # Search for patterns like _1.jpg, _2.jpg, etc.
            import re
            match = re.search(r'_(\d+)\.(jpg|jpeg|png|bmp|tiff)$', filename.lower())
            if match:
                return int(match.group(1))
            
            # If not found, return 0 to maintain original order
            return 0
        except:
            return 0
    
    def _detect_class_names(self, data_dir: Path) -> List[str]:
        """
        Automatically detect class names based on subdirectories.
        
        Args:
            data_dir: Directory with data
            
        Returns:
            List of class names
        """
        # Search for subdirectories containing images
        class_dirs = []
        for item in data_dir.iterdir():
            if item.is_dir():
                # Check if directory contains images
                has_images = any(
                    item.glob(ext) for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                )
                if has_images:
                    class_dirs.append(item.name)
        
        # Sort for consistency
        class_dirs.sort()
        
        if not class_dirs:
            print(f"Warning: No classes found in {data_dir}")
            return ['Regular', 'Peculiar']  # Fallback
        
        print(f"Classes auto-detected: {class_dirs}")
        return class_dirs
    
    def create_mosaic_for_dataset(
        self,
        dataset_name: str,
        data_dir: Path = None,
        class_names: List[str] = None
    ) -> Path:
        """
        Create mosaic for a specific dataset.
        
        Args:
            dataset_name: Dataset name (e.g., 'sdss', 'splus')
            data_dir: Directory with data (default: data/complete_{dataset_name})
            class_names: Class names (if None, auto-detects)
            
        Returns:
            Path to saved mosaic
        """
        if data_dir is None:
            data_dir = Path(f"data/images/{dataset_name}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        # Auto-detect classes if not specified
        if class_names is None:
            class_names = self._detect_class_names(data_dir)
        
        return self.create_mosaic(
            data_dir=data_dir,
            class_names=class_names,
            mosaic_name=f"{dataset_name}_mosaic"
        )
    
    def create_all_mosaics(self, base_data_dir: Path = Path("data")) -> List[Path]:
        """
        Create mosaics for all datasets found.
        
        Args:
            base_data_dir: Base directory with data
            
        Returns:
            List of paths to created mosaics
        """
        mosaics_created = []
        
        # Search for complete_* directories
        for complete_dir in base_data_dir.glob("complete_*"):
            if complete_dir.is_dir():
                dataset_name = complete_dir.name.replace("complete_", "")
                print(f"Processing dataset: {dataset_name}")
                
                try:
                    mosaic_path = self.create_mosaic_for_dataset(dataset_name, complete_dir)
                    mosaics_created.append(mosaic_path)
                except Exception as e:
                    print(f"Error creating mosaic for {dataset_name}: {e}")
        
        return mosaics_created
    
    def create_comparison_mosaic(
        self,
        sdss_dir: Path,
        splus_dir: Path,
        class_names: List[str] = None
    ) -> Path:
        """
        Create comparative mosaic between SDSS and S-PLUS.
        
        Args:
            sdss_dir: Directory with SDSS data
            splus_dir: Directory with S-PLUS data
            class_names: Class names
            
        Returns:
            Path to comparison mosaic
        """
        print("Creating SDSS vs S-PLUS comparison mosaic")
        
        # Auto-detect classes if not specified
        if class_names is None:
            sdss_classes = self._detect_class_names(sdss_dir)
            splus_classes = self._detect_class_names(splus_dir)
            # Use SDSS classes as reference
            class_names = sdss_classes
        
        # Load images from both datasets
        sdss_images = self._load_images_by_class(sdss_dir, class_names, 5)
        splus_images = self._load_images_by_class(splus_dir, class_names, 5)
        
        # Create comparison figure
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 5, figure=fig, hspace=0.1, wspace=0.1)
        
        # fig.suptitle('SDSS vs S-PLUS Comparison', fontsize=16, fontweight='bold', y=0.95)  # Title commented
        
        # Rows 1-2: SDSS Regular
        # Rows 3-4: S-PLUS Regular
        # Rows 5-6: SDSS Peculiar
        # Rows 7-8: S-PLUS Peculiar
        
        datasets = [('SDSS', sdss_images), ('S-PLUS', splus_images)]
        row_idx = 0
        
        for dataset_name, images_by_class in datasets:
            for class_name in class_names:
                class_images = images_by_class[class_name]
                
                for col in range(self.cols):
                    ax = fig.add_subplot(gs[row_idx, col])
                    
                    if col < len(class_images):
                        img, filename = class_images[col]
                        img_resized = img.resize(self.image_size, Image.Resampling.LANCZOS)
                        ax.imshow(img_resized)
                        
                        # Galaxy name
                        if "augmented" in str(sdss_dir) or "complete" in str(sdss_dir):
                            # For data augmentation mosaics: simple name at bottom right
                            simple_name = filename.split('_aug_')[0] if '_aug_' in filename else filename.split('_')[0] + '_' + filename.split('_')[1]
                            clean_name = simple_name.replace('_', '')
                            ax.text(0.95, 0.05, clean_name,
                                   transform=ax.transAxes,
                                   fontsize=8,
                                   color='white',
                                   ha='right',
                                   va='bottom')
                        else:
                            # For original mosaics: clean name at bottom right
                            clean_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', '')
                            ax.text(0.95, 0.05, clean_name,
                                   transform=ax.transAxes,
                                   fontsize=8,
                                   color='white',
                                   ha='right',
                                   va='bottom')
                        
                        # Class and dataset label
                        if col == 0:
                            label = f"{dataset_name}\n{class_name}"
                            ax.text(-0.15, 0.5, label,
                                   transform=ax.transAxes,
                                   fontsize=10,
                                   fontweight='bold',
                                   color='black',
                                   verticalalignment='center',
                                   horizontalalignment='right',
                                   rotation=90)
                    else:
                        ax.imshow(np.ones((*self.image_size, 3)) * 0.9, cmap='gray')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                
                row_idx += 1
        
        # Save comparison mosaic
        mosaic_path = self.output_dir / "comparison_mosaic.png"
        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.02, wspace=0.0001, hspace=0.0001)
        plt.savefig(mosaic_path, dpi=300,
                   facecolor='white', edgecolor='none', pad_inches=0.05)
        plt.close()
        
        print(f"Comparison mosaic saved to: {mosaic_path}")
        return mosaic_path
    
    def create_unique_augmentation_mosaic(self) -> Optional[Path]:
        """
        Create 4 separate mosaics: one galaxy per class per dataset.
        Each mosaic has 1 row x 5 columns with data augmentation variations.
        
        Returns:
            Path to last created mosaic or None if no data found
        """
        # Create 4 separate mosaics: SDSS-REG, SDSS-IRR_PEC, S-PLUS-REG, S-PLUS-IRR_PEC
        created_mosaics = []
        
        # 1. SDSS - REG mosaic
        sdss_dir = Path("data/complete_sdss")
        if sdss_dir.exists():
            reg_mosaic = self._create_single_class_mosaic(sdss_dir, "SDSS", "reg")
            if reg_mosaic:
                created_mosaics.append(reg_mosaic)
        
        # 2. SDSS - IRR_PEC mosaic
        if sdss_dir.exists():
            irr_pec_mosaic = self._create_single_class_mosaic(sdss_dir, "SDSS", "irr_pec")
            if irr_pec_mosaic:
                created_mosaics.append(irr_pec_mosaic)
        
        # 3. S-PLUS - REG mosaic
        splus_dir = Path("data/complete_splus")
        if splus_dir.exists():
            reg_mosaic = self._create_single_class_mosaic(splus_dir, "S-PLUS", "reg")
            if reg_mosaic:
                created_mosaics.append(reg_mosaic)
        
        # 4. S-PLUS - IRR_PEC mosaic
        if splus_dir.exists():
            irr_pec_mosaic = self._create_single_class_mosaic(splus_dir, "S-PLUS", "irr_pec")
            if irr_pec_mosaic:
                created_mosaics.append(irr_pec_mosaic)
        
        if not created_mosaics:
            print("No data augmentation mosaics created")
            return None
        
        print(f"{len(created_mosaics)} data augmentation mosaics created")
        return created_mosaics[-1]  # Return last created
    
    def _create_single_class_mosaic(self, dataset_dir: Path, dataset_name: str, class_name: str) -> Optional[Path]:
        """
        Create mosaic for a single class of a dataset.
        
        Args:
            dataset_dir: Dataset directory
            dataset_name: Dataset name (SDSS or S-PLUS)
            class_name: Class name (reg or irr_pec)
            
        Returns:
            Path to created mosaic or None
        """
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"Class {class_name} not found in {dataset_name}")
            return None
        
        # Get all images from this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(class_dir.glob(ext))
        
        if not image_files:
            print(f"No images found for {class_name} in {dataset_name}")
            return None
        
        # Search for a galaxy with multiple variations
        selected_galaxy = None
        galaxy_variations = []
        
        # Try random galaxies until finding one with 5+ variations
        for _ in range(10):  # Try up to 10 times
            base_image = random.choice(image_files)
            base_name = base_image.stem
            galaxy_id = base_name.split('_')[0] + '_' + base_name.split('_')[1]
            
            variations = []
            for img_file in image_files:
                if galaxy_id in img_file.stem:
                    variations.append(img_file)
            
            if len(variations) >= 5:
                selected_galaxy = galaxy_id
                galaxy_variations = variations[:5]
                print(f"Galaxy {dataset_name}-{class_name} selected: {galaxy_id} with {len(variations)} variations")
                break
        
        if not selected_galaxy:
            print(f"No galaxy with 5 variations found for {class_name} in {dataset_name}")
            return None
        
        # Sort variations by augmentation number
        sorted_variations = sorted(galaxy_variations, key=lambda x: self._extract_aug_number(x.name))
        
        # Load images
        images = []
        file_names = []
        for img_file in sorted_variations:
            try:
                img = Image.open(img_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                file_names.append(img_file.name)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        if len(images) < 5:
            print(f"Only {len(images)} images loaded for {dataset_name}-{class_name}")
            return None
        
        # Create 1x5 mosaic (one row with 5 columns)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        for col, (img, filename) in enumerate(zip(images, file_names)):
            ax = axes[col]
            
            # Resize image
            img_resized = img.resize((200, 200), Image.Resampling.LANCZOS)
            ax.imshow(img_resized)
            
            # Add information in last image (column 5)
            if col == 4:  # Last column (index 4)
                simple_name = filename.split('_aug_')[0] if '_aug_' in filename else filename.split('_')[0] + '_' + filename.split('_')[1]
                clean_name = simple_name.replace('_', '')
                info_text = f"{clean_name}\n{class_name.upper()}\n{dataset_name}"
                ax.text(0.95, 0.05, info_text,
                       transform=ax.transAxes,
                       fontsize=8,
                       color='white',
                       ha='right',
                       va='bottom')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        
        # Save mosaic
        output_path = self.output_dir / f"unique_augmentation_{dataset_name.lower()}_{class_name}_mosaic.png"
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
        plt.savefig(output_path, dpi=150, pad_inches=0.05)
        plt.close()
        
        print(f"Mosaic {dataset_name}-{class_name} saved to: {output_path}")
        return output_path


def create_mosaics_for_project():
    """
    Utility function to create all project mosaics.
    Creates specific mosaics as requested:
    1. Original SDSS mosaic
    2. Original S-PLUS mosaic
    3. SDSS vs S-PLUS comparison mosaic
    4. SDSS data augmentation mosaic
    5. S-PLUS data augmentation mosaic
    6. 4 unique data augmentation images mosaic
    """
    generator = MosaicGenerator()
    
    print("Creating specific mosaics for Galaxy Classification project...")
    mosaics = []
    
    # 1. Original SDSS mosaic
    sdss_dir = Path("data/images/sdss")
    if sdss_dir.exists():
        print("1. Creating original SDSS mosaic...")
        sdss_mosaic = generator.create_mosaic_for_dataset("sdss", sdss_dir)
        mosaics.append(sdss_mosaic)
    
    # 2. Original S-PLUS mosaic
    splus_dir = Path("data/images/splus")
    if splus_dir.exists():
        print("2. Creating original S-PLUS mosaic...")
        splus_mosaic = generator.create_mosaic_for_dataset("splus", splus_dir)
        mosaics.append(splus_mosaic)
    
    # 3. SDSS vs S-PLUS comparison mosaic
    if sdss_dir.exists() and splus_dir.exists():
        print("3. Creating SDSS vs S-PLUS comparison mosaic...")
        comparison_mosaic = generator.create_comparison_mosaic(sdss_dir, splus_dir)
        mosaics.append(comparison_mosaic)
    
    # 4. SDSS data augmentation mosaic
    sdss_aug_dir = Path("data/complete_sdss")
    if sdss_aug_dir.exists():
        print("4. Creating SDSS data augmentation mosaic...")
        sdss_aug_mosaic = generator.create_mosaic_for_dataset("augmented_sdss", sdss_aug_dir)
        mosaics.append(sdss_aug_mosaic)
    
    # 5. S-PLUS data augmentation mosaic
    splus_aug_dir = Path("data/complete_splus")
    if splus_aug_dir.exists():
        print("5. Creating S-PLUS data augmentation mosaic...")
        splus_aug_mosaic = generator.create_mosaic_for_dataset("augmented_splus", splus_aug_dir)
        mosaics.append(splus_aug_mosaic)
    
    # 6. 4 unique data augmentation images mosaic
    print("6. Creating 4 unique data augmentation images mosaic...")
    unique_aug_mosaic = generator.create_unique_augmentation_mosaic()
    if unique_aug_mosaic:
        mosaics.append(unique_aug_mosaic)
    
    print(f"\nSuccess: {len(mosaics)} mosaics created:")
    for i, mosaic in enumerate(mosaics, 1):
        print(f"  {i}. {mosaic}")
    
    return mosaics


if __name__ == "__main__":
    # Usage example
    create_mosaics_for_project()
