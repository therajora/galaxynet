#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# Default variables
CONFIG_FILE=""
DEVICE="auto"
SEED=42
VALIDATE_ONLY=false
MODEL_NAME=""

# Function to display help
show_help() {
    echo -e "${BLUE}=== Pre-trained Model Training Script (Kaggle) ===${NC}"
    echo -e "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -c, --config FILE      JSON configuration file (required)"
    echo "  -d, --device DEVICE    Device: auto, cuda, cpu (default: ${DEVICE})"
    echo "  -s, --seed NUM         Seed for reproducibility (default: ${SEED})"
    echo "  -m, --model NAME       Model name: efficientnet_b0, resnet50_v1, vit_b_16, etc."
    echo "  --validate-only        Only validate configuration without training"
    echo "  -h, --help             Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $(basename "$0") -c model/pretrained/configs/kaggle/efficientnet_b0_kaggle.json"
    echo "  $(basename "$0") -m efficientnet_b0 -d cuda -s 123"
    echo "  $(basename "$0") -c model/pretrained/configs/kaggle/resnet50_v1_kaggle.json --validate-only"
    echo "  $(basename "$0") -m vit_b_16 -d cuda"
    echo ""
    echo "AVAILABLE CONFIGURATIONS:"
    echo ""
    echo "EFFICIENTNET FAMILY:"
    echo "  - efficientnet_b0_kaggle.json       - EfficientNet-B0"
    echo "  - efficientnet_b1_kaggle.json       - EfficientNet-B1"
    echo "  - efficientnet_b2_kaggle.json       - EfficientNet-B2"
    echo "  - efficientnet_b3_kaggle.json       - EfficientNet-B3"
    echo "  - efficientnet_b4_kaggle.json       - EfficientNet-B4"
    echo ""
    echo "RESNET FAMILY:"
    echo "  - resnet50_v1_kaggle.json           - ResNet-50v1"
    echo "  - resnet50_v2_kaggle.json           - ResNet-50v2"
    echo "  - resnext50_kaggle.json             - ResNeXt-50"
    echo ""
    echo "VISION TRANSFORMER FAMILY:"
    echo "  - vit_b_16_kaggle.json              - ViT-Base/16"
    echo "  - vit_b_32_kaggle.json              - ViT-Base/32"
    echo ""
    echo "NOTE: This script is optimized for Kaggle environment (without conda)"
}

# Function to get configuration file based on model name
get_config_file() {
    local model_name="$1"
    case "$model_name" in
        # EfficientNet
        "efficientnet_b0")
            echo "model/pretrained/configs/kaggle/efficientnet_b0_kaggle.json"
            ;;
        "efficientnet_b1")
            echo "model/pretrained/configs/kaggle/efficientnet_b1_kaggle.json"
            ;;
        "efficientnet_b2")
            echo "model/pretrained/configs/kaggle/efficientnet_b2_kaggle.json"
            ;;
        "efficientnet_b3")
            echo "model/pretrained/configs/kaggle/efficientnet_b3_kaggle.json"
            ;;
        "efficientnet_b4")
            echo "model/pretrained/configs/kaggle/efficientnet_b4_kaggle.json"
            ;;
        # ResNet
        "resnet50_v1")
            echo "model/pretrained/configs/kaggle/resnet50_v1_kaggle.json"
            ;;
        "resnet50_v2")
            echo "model/pretrained/configs/kaggle/resnet50_v2_kaggle.json"
            ;;
        "resnext50")
            echo "model/pretrained/configs/kaggle/resnext50_kaggle.json"
            ;;
        # Vision Transformer
        "vit_b_16")
            echo "model/pretrained/configs/kaggle/vit_b_16_kaggle.json"
            ;;
        "vit_b_32")
            echo "model/pretrained/configs/kaggle/vit_b_32_kaggle.json"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Process arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) CONFIG_FILE="$2"; shift 2 ;;
        -d|--device) DEVICE="$2"; shift 2 ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        -m|--model) MODEL_NAME="$2"; shift 2 ;;
        --validate-only) VALIDATE_ONLY=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo -e "${RED}Error: Unknown option '$1'${NC}"; show_help; exit 1 ;;
    esac
done

# If model was specified, use corresponding configuration file
if [ -n "$MODEL_NAME" ]; then
    if [ -z "$CONFIG_FILE" ]; then
        CONFIG_FILE=$(get_config_file "$MODEL_NAME")
        if [ -z "$CONFIG_FILE" ]; then
            echo -e "${RED}Error: Model '$MODEL_NAME' not recognized.${NC}"
            echo -e "${YELLOW}Available models: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, resnet50_v1, resnet50_v2, resnext50, vit_b_16, vit_b_32${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Warning: Both model (-m) and configuration file (-c) were specified. Using configuration file.${NC}"
    fi
fi

# Check if configuration file was specified
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not specified.${NC}"
    echo -e "${YELLOW}Use -c to specify a file or -m to use a pre-configured model.${NC}"
    show_help
    exit 1
fi

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if PyTorch is available
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Warning: PyTorch not found. Installing dependencies...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Check other dependencies
python -c "
import sys
missing = []
try:
    import transformers
except ImportError:
    missing.append('transformers')
try:
    import datasets
except ImportError:
    missing.append('datasets')
try:
    import huggingface_hub
except ImportError:
    missing.append('huggingface_hub')
try:
    import albumentations
except ImportError:
    missing.append('albumentations')
try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')
try:
    import seaborn
except ImportError:
    missing.append('seaborn')
try:
    import pandas
except ImportError:
    missing.append('pandas')
try:
    import sklearn
except ImportError:
    missing.append('scikit-learn')
try:
    import tabulate
except ImportError:
    missing.append('tabulate')

if missing:
    print(f'Installing missing dependencies: {missing}')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)
    print('Dependencies installed successfully!')
else:
    print('All dependencies are available!')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error installing dependencies${NC}"
    exit 1
fi

echo -e "${BLUE}Executing pre-trained model training (Kaggle)...${NC}"

# Set environment variables for Kaggle
export PYTHONPATH="/kaggle/working/galaxynet:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"

# Execute training
python -c "
import sys
import os
sys.path.insert(0, '/kaggle/working/galaxynet')

# Import and execute training
from model.pretrained.train_pretrained import main as train_main
import argparse

# Create arguments
args = argparse.Namespace()
args.config_file = '$CONFIG_FILE'
args.device = '$DEVICE'
args.seed = $SEED
args.validate_only = $VALIDATE_ONLY

print(f'Configuration: {args.config_file}')
print(f'Device: {args.device}')
print(f'Seed: {args.seed}')
print(f'Validate only: {args.validate_only}')

# Execute training
try:
    train_main(args)
    print('Training completed successfully!')
except Exception as e:
    print(f'Error during training: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${BLUE}Results saved in: /kaggle/working/galaxynet/results/${NC}"
else
    echo -e "${RED}Error during training${NC}"
    exit 1
fi
