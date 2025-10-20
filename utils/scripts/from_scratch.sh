#!/bin/bash

# Script for training CNN models from scratch
# Module: from_scratch

set -e  # Stop script on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to show help
show_help() {
    echo "=== From Scratch Training Script ==="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -e, --epochs NUM        Number of epochs (default: 10)"
    echo "  -s, --samples NUM       Number of samples for prediction (default: 5)"
    echo "  -d, --data-dir DIR      Data directory (default: data/complete_sdss)"
    echo "  -m, --model-type TYPE   Model type: simple, v2 (default: simple)"
    echo "  -b, --batch-size NUM    Batch size (default: 32)"
    echo "  -l, --learning-rate NUM Learning rate (default: 0.001)"
    echo "  -n, --model-name NAME   Model name (default: galaxy_cnn)"
    echo "  --dataset-name NAME     Dataset name (default: complete_sdss)"
    echo "  --description TEXT      Version description"
    echo "  --evaluate              Evaluate model after training"
    echo "  --no-versioning         Disable versioning"
    echo "  --device DEVICE         Device: cuda, cpu, auto (default: auto)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -e 20 -s 10                    # 20 epochs, 10 samples"
    echo "  $0 -e 50 -m v2 -b 64             # 50 epochs, v2 model, batch 64"
    echo "  $0 -d data/complete_splus -e 15   # Use S-PLUS data, 15 epochs"
    echo "  $0 --evaluate -e 30               # 30 epochs with evaluation"
    echo ""
}

# Default values
EPOCHS=10
SAMPLES=5
DATA_DIR="data/complete_sdss"
MODEL_TYPE="simple"
BATCH_SIZE=32
LEARNING_RATE=0.001
MODEL_NAME="galaxy_cnn"
DATASET_NAME="complete_sdss"
DESCRIPTION=""
EVALUATE=false
USE_VERSIONING=true
DEVICE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -m|--model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --description)
            DESCRIPTION="$2"
            shift 2
            ;;
        --evaluate)
            EVALUATE=true
            shift
            ;;
        --no-versioning)
            USE_VERSIONING=false
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validations
if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
    print_error "Number of epochs must be a positive integer"
    exit 1
fi

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || [ "$SAMPLES" -lt 0 ]; then
    print_error "Number of samples must be a non-negative integer"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    print_error "Batch size must be a positive integer"
    exit 1
fi

if [[ "$MODEL_TYPE" != "simple" && "$MODEL_TYPE" != "v2" ]]; then
    print_error "Model type must be 'simple' or 'v2'"
    exit 1
fi

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" && "$DEVICE" != "auto" ]]; then
    print_error "Device must be 'cuda', 'cpu' or 'auto'"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    print_info "Run the augmentation pipeline first to generate the data"
    exit 1
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "Conda environment not detected. Activating galaxynet..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate galaxynet
fi

print_info "Conda environment: $CONDA_DEFAULT_ENV"

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

print_info "Project directory: $PROJECT_DIR"

# Build command
CMD="python model/from_scratch/train_from_scratch.py"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --num_epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --model_type $MODEL_TYPE"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --device $DEVICE"
CMD="$CMD --predict_samples $SAMPLES"

if [ "$EVALUATE" = true ]; then
    CMD="$CMD --evaluate"
fi

if [ "$USE_VERSIONING" = false ]; then
    CMD="$CMD --no-versioning"
fi

if [ -n "$DESCRIPTION" ]; then
    CMD="$CMD --description \"$DESCRIPTION\""
fi

# Show configurations
echo ""
print_info "=== TRAINING CONFIGURATION ==="
echo "Epochs: $EPOCHS"
echo "Samples for prediction: $SAMPLES"
echo "Data directory: $DATA_DIR"
echo "Model type: $MODEL_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Model name: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Device: $DEVICE"
echo "Versioning: $USE_VERSIONING"
echo "Evaluation: $EVALUATE"
if [ -n "$DESCRIPTION" ]; then
    echo "Description: $DESCRIPTION"
fi
echo ""

# Confirm execution
read -p "Do you want to continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training cancelled by user"
    exit 0
fi

# Execute training
print_info "Starting training..."
echo "Command: $CMD"
echo ""

# Capture start time
START_TIME=$(date +%s)

# Execute command
if eval $CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "Training completed successfully!"
    print_info "Total time: ${DURATION}s"
    
    # Show information about results
    if [ "$USE_VERSIONING" = true ]; then
        echo ""
        print_info "=== RESULTS INFORMATION ==="
        echo "Models saved in: results/models/"
        echo "Logs saved in: results/logs/"
        echo "Metrics saved in: results/metrics/"
        echo ""
        print_info "To list versions: python results/manage_versions.py list"
        print_info "To see details: python results/manage_versions.py show <version_id>"
    else
        echo ""
        print_info "=== RESULTS INFORMATION ==="
        echo "Models saved in: model_artifacts/"
    fi
    
else
    print_error "Error during training"
    exit 1
fi

print_success "Script completed!"
