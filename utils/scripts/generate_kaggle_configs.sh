#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# Function to display help
show_help() {
    echo -e "${BLUE}=== Kaggle Configuration Generator ===${NC}"
    echo -e "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -m, --model MODEL       Model name (or 'all' for all models)"
    echo "  -o, --output-dir DIR    Output directory (default: model/pretrained/configs/kaggle)"
    echo "  --epochs NUM            Number of epochs"
    echo "  --batch-size NUM        Batch size"
    echo "  --learning-rate NUM     Learning rate"
    echo "  --repo-name NAME        Hugging Face repository name"
    echo "  -h, --help              Show this help"
    echo ""
    echo "AVAILABLE MODELS:"
    echo "  efficientnet_b0         - EfficientNet-B0"
    echo "  efficientnet_b1         - EfficientNet-B1"
    echo "  efficientnet_b2         - EfficientNet-B2"
    echo "  efficientnet_b3         - EfficientNet-B3"
    echo "  efficientnet_b4         - EfficientNet-B4"
    echo "  resnet50_v1             - ResNet-50v1"
    echo "  resnet50_v2             - ResNet-50v2"
    echo "  resnext50               - ResNeXt-50"
    echo "  vit_b_16                - ViT-Base/16"
    echo "  vit_b_32                - ViT-Base/32"
    echo "  all                     - All models"
    echo ""
    echo "EXAMPLES:"
    echo "  $(basename "$0") --model all"
    echo "  $(basename "$0") --model efficientnet_b0 --epochs 15 --batch-size 64"
    echo "  $(basename "$0") --model resnet50_v1 --learning-rate 0.0005"
    echo "  $(basename "$0") --model all --repo-name my-repo/galaxy-dataset"
    echo ""
}

# Default variables
MODEL="all"
OUTPUT_DIR="model/pretrained/configs/kaggle"
EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
REPO_NAME=""

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --repo-name)
            REPO_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Build Python command
PYTHON_CMD="python utils/scripts/generate_kaggle_configs.py --model $MODEL --output-dir $OUTPUT_DIR"

if [[ -n "$EPOCHS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --epochs $EPOCHS"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
fi

if [[ -n "$LEARNING_RATE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --learning-rate $LEARNING_RATE"
fi

if [[ -n "$REPO_NAME" ]]; then
    PYTHON_CMD="$PYTHON_CMD --repo-name $REPO_NAME"
fi

# Execute command
echo -e "${BLUE}Generating configurations for Kaggle...${NC}"
echo -e "${YELLOW}Command: $PYTHON_CMD${NC}"
echo ""

eval $PYTHON_CMD

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}Configurations generated successfully!${NC}"
    echo -e "${BLUE}Location: $OUTPUT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}To use in Kaggle:${NC}"
    echo "  bash utils/scripts/train_pretrained.sh -c $OUTPUT_DIR/${MODEL}_kaggle.json"
else
    echo -e "${RED}Error generating configurations${NC}"
    exit 1
fi
