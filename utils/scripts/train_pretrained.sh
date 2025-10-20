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
    echo -e "${BLUE}=== Pre-trained Model Training Script ===${NC}"
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
    echo "  $(basename "$0") -c model/pretrained/configs/efficientnet_b0.json"
    echo "  $(basename "$0") -m efficientnet_b0 -d cuda -s 123"
    echo "  $(basename "$0") -c model/pretrained/configs/resnet50_v1.json --validate-only"
    echo "  $(basename "$0") -m vit_b_16 -d cuda"
    echo ""
    echo "AVAILABLE CONFIGURATIONS:"
    echo ""
    echo "EFFICIENTNET FAMILY:"
    echo "  - efficientnet_b0.json       - EfficientNet-B0"
    echo "  - efficientnet_b1.json       - EfficientNet-B1"
    echo "  - efficientnet_b2.json       - EfficientNet-B2"
    echo "  - efficientnet_b3.json       - EfficientNet-B3"
    echo "  - efficientnet_b4.json       - EfficientNet-B4"
    echo ""
    echo "RESNET FAMILY:"
    echo "  - resnet50_v1.json           - ResNet-50v1"
    echo "  - resnet50_v2.json           - ResNet-50v2"
    echo "  - resnext50.json             - ResNeXt-50"
    echo ""
    echo "VISION TRANSFORMER FAMILY:"
    echo "  - vit_b_16.json              - ViT-Base/16"
    echo "  - vit_b_32.json              - ViT-Base/32"
    echo ""
    echo "ADDITIONAL EXAMPLES:"
    echo "  - resnet50_huggingface.json  - ResNet50 with Hugging Face data"
    echo "  - resnet50_cifar10.json      - ResNet50 with CIFAR-10 (HF example)"
    echo ""
}

# Function to map model name to configuration file
get_config_file() {
    local model_name="$1"
    case "$model_name" in
        # EfficientNet
        "efficientnet_b0")
            echo "model/pretrained/configs/efficientnet_b0.json"
            ;;
        "efficientnet_b1")
            echo "model/pretrained/configs/efficientnet_b1.json"
            ;;
        "efficientnet_b2")
            echo "model/pretrained/configs/efficientnet_b2.json"
            ;;
        "efficientnet_b3")
            echo "model/pretrained/configs/efficientnet_b3.json"
            ;;
        "efficientnet_b4")
            echo "model/pretrained/configs/efficientnet_b4.json"
            ;;
        # ResNet
        "resnet50_v1")
            echo "model/pretrained/configs/resnet50_v1.json"
            ;;
        "resnet50_v2")
            echo "model/pretrained/configs/resnet50_v2.json"
            ;;
        "resnext50")
            echo "model/pretrained/configs/resnext50.json"
            ;;
        # Vision Transformer
        "vit_b_16")
            echo "model/pretrained/configs/vit_b_16.json"
            ;;
        "vit_b_32")
            echo "model/pretrained/configs/vit_b_32.json"
            ;;
        # Shortcuts for compatibility
        "resnet50")
            echo "model/pretrained/configs/resnet50_v1.json"
            ;;
        "efficientnet")
            echo "model/pretrained/configs/efficientnet_b0.json"
            ;;
        "vit")
            echo "model/pretrained/configs/vit_b_16.json"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
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
            echo -e "${YELLOW}Available models: resnet50, efficientnet, vit${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Warning: Both --model and --config were specified. Using --config.${NC}"
    fi
fi

# Configuration file validation
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file is required.${NC}"
    echo -e "${YELLOW}Use --config or --model to specify the configuration.${NC}"
    show_help
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Device validation
if [[ "$DEVICE" != "auto" && "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
    echo -e "${RED}Error: --device must be 'auto', 'cuda' or 'cpu'.${NC}"
    exit 1
fi

# Seed validation
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: --seed must be an integer.${NC}"
    exit 1
fi

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}Executing pre-trained model training...${NC}"

# Activate conda environment (if not active)
if ! conda info --envs | grep -q "galaxynet"; then
    echo -e "${RED}Error: Conda environment 'galaxynet' not found.${NC}"
    exit 1
fi

if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "galaxynet" ]]; then
    echo -e "${BLUE}Activating conda environment galaxynet...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate galaxynet
    echo -e "${GREEN}Environment galaxynet activated${NC}"
fi

echo -e "${BLUE}=== TRAINING CONFIGURATION ===${NC}"
echo -e "Configuration file: ${CONFIG_FILE}"
echo -e "Device: ${DEVICE}"
echo -e "Seed: ${SEED}"
echo -e "Validate only: ${VALIDATE_ONLY}"
echo -e "${BLUE}=======================================${NC}"

# Ask for confirmation if not validation only
if [ "$VALIDATE_ONLY" = false ]; then
    read -p "$(echo -e "${YELLOW}Do you want to start training with these configurations? (y/N): ${NC}")" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Training cancelled.${NC}"
        exit 0
    fi
fi

START_TIME=$(date +%s)

# Build Python command
PYTHON_CMD="python model/pretrained/train_pretrained.py"
PYTHON_CMD+=" --config \"$CONFIG_FILE\""
PYTHON_CMD+=" --device \"$DEVICE\""
PYTHON_CMD+=" --seed $SEED"

if [ "$VALIDATE_ONLY" = true ]; then
    PYTHON_CMD+=" --validate-only"
fi

echo -e "${BLUE}Starting process...${NC}"
echo -e "${BLUE}Command: ${PYTHON_CMD}${NC}"

# Execute Python script
if eval "$PYTHON_CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ "$VALIDATE_ONLY" = true ]; then
        echo -e "${GREEN}Validation completed successfully!${NC}"
    else
        echo -e "${GREEN}Training completed successfully!${NC}"
    fi
    echo -e "${BLUE}Total time: ${DURATION}s${NC}"
else
    echo -e "${RED}Error during execution${NC}"
    exit 1
fi

if [ "$VALIDATE_ONLY" = false ]; then
    echo -e "${BLUE}=== RESULTS INFORMATION ===${NC}"
    echo -e "Models saved in: results/models/"
    echo -e "Logs saved in: results/logs/"
    echo -e "Plots saved in: results/models/"
    echo ""
    echo -e "To list versions: ${BLUE}python results/manage_versions.py list${NC}"
    echo -e "To see details: ${BLUE}python results/manage_versions.py show <version_id>${NC}"
fi

echo -e "${GREEN}Script completed!${NC}"
