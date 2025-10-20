#!/bin/bash

# Script to execute data augmentation and balancing
# Uses conda environment galaxynet

echo "=== Data Augmentation Script ==="
echo

# Activate conda environment
echo "Activating conda environment galaxynet..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate galaxynet

# Check if environment was activated
if [[ "$CONDA_DEFAULT_ENV" != "galaxynet" ]]; then
    echo "ERROR: Could not activate galaxynet environment"
    echo "Execute: conda activate galaxynet"
    exit 1
fi

echo "Environment galaxynet activated"
echo

# Project directory
PROJECT_DIR="/home/rafael/research/galaxy-classification-sdss-splus"
cd "$PROJECT_DIR"

# Check if directories exist
if [ ! -d "data/images/sdss" ]; then
    echo "ERROR: Directory data/images/sdss not found"
    exit 1
fi

if [ ! -d "data/images/splus" ]; then
    echo "ERROR: Directory data/images/splus not found"
    exit 1
fi

echo "Data directories found"
echo

# Options menu
echo "Choose an option:"
echo "1) Analyze SDSS data"
echo "2) Analyze S-PLUS data"
echo "3) Augment SDSS data"
echo "4) Augment S-PLUS data"
echo "5) Balance SDSS data"
echo "6) Balance S-PLUS data"
echo "7) Complete SDSS pipeline"
echo "8) Complete S-PLUS pipeline"
echo "9) Exit"
echo

read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo "Analyzing SDSS data..."
        python run_augmentation.py -i data/images/sdss --mode analyze
        ;;
    2)
        echo "Analyzing S-PLUS data..."
        python run_augmentation.py -i data/images/splus --mode analyze
        ;;
    3)
        echo "Augmenting SDSS data..."
        python run_augmentation.py -i data/images/sdss -o data/augmented_sdss -s sdss --mode augment
        ;;
    4)
        echo "Augmenting S-PLUS data..."
        python run_augmentation.py -i data/images/splus -o data/augmented_splus -s splus --mode augment
        ;;
    5)
        echo "Balancing SDSS data..."
        python run_augmentation.py -i data/images/sdss -o data/balanced_sdss --mode balance
        ;;
    6)
        echo "Balancing S-PLUS data..."
        python run_augmentation.py -i data/images/splus -o data/balanced_splus --mode balance
        ;;
    7)
        echo "Executing complete SDSS pipeline..."
        python run_augmentation.py -i data/images/sdss -o data/complete_sdss -s sdss --mode complete
        ;;
    8)
        echo "Executing complete S-PLUS pipeline..."
        python run_augmentation.py -i data/images/splus -o data/complete_splus -s splus --mode complete
        ;;
    9)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo
echo "Process completed!"
echo "Check the results in the output directories."
