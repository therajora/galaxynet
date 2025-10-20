#!/bin/bash

# Script for uploading datasets to Hugging Face Hub
# Module: datasets

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
    echo "=== Hugging Face Hub Upload Script ==="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --data-dir DIR      Data directory (required)"
    echo "  -r, --repo-id ID        Repository ID (ex: username/dataset-name)"
    echo "  -s, --split RATIO       Train/test split ratio (0.0-1.0)"
    echo "  -b, --binary            Create binary dataset (Regular vs Peculiar)"
    echo "  -p, --private           Private repository"
    echo "  -t, --token TOKEN       Hugging Face token"
    echo "  -m, --message MSG       Commit message"
    echo "  --create-only           Only create local dataset (no upload)"
    echo "  --metadata-path PATH    Path to metadata"
    echo "  -h, --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -d data/complete_sdss -r username/galaxy-dataset"
    echo "  $0 -d data/complete_sdss -r username/galaxy-dataset -s 0.2"
    echo "  $0 -d data/complete_sdss -r username/galaxy-binary -b"
    echo "  $0 -d data/complete_sdss --create-only"
    echo "  $0 -d data/complete_sdss -r username/galaxy-dataset -p -t hf_xxx"
    echo ""
}

# Default values
DATA_DIR=""
REPO_ID=""
SPLIT=0.0
BINARY=false
PRIVATE=false
TOKEN=""
MESSAGE="Upload galaxy dataset"
CREATE_ONLY=false
METADATA_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -r|--repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -b|--binary)
            BINARY=true
            shift
            ;;
        -p|--private)
            PRIVATE=true
            shift
            ;;
        -t|--token)
            TOKEN="$2"
            shift 2
            ;;
        -m|--message)
            MESSAGE="$2"
            shift 2
            ;;
        --create-only)
            CREATE_ONLY=true
            shift
            ;;
        --metadata-path)
            METADATA_PATH="$2"
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
if [ -z "$DATA_DIR" ]; then
    print_error "Data directory is required (-d/--data-dir)"
    show_help
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

if [ "$CREATE_ONLY" = false ] && [ -z "$REPO_ID" ]; then
    print_error "Repository ID is required when not using --create-only"
    show_help
    exit 1
fi

if ! [[ "$SPLIT" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$SPLIT < 0" | bc -l) )) || (( $(echo "$SPLIT >= 1" | bc -l) )); then
    print_error "Split must be between 0 and 1"
    exit 1
fi

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Load .env file if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    print_info "Loading environment variables from .env file..."
    source "$PROJECT_DIR/.env"
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "Conda environment not detected. Activating galaxynet..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate galaxynet
fi

print_info "Conda environment: $CONDA_DEFAULT_ENV"

print_info "Project directory: $PROJECT_DIR"

# Check dependencies
print_info "Checking dependencies..."
if ! python -c "import datasets, huggingface_hub" 2>/dev/null; then
    print_warning "Hugging Face dependencies not found. Installing..."
    pip install datasets huggingface_hub
fi

# Build command
CMD="python model/datasets/upload_dataset.py"
CMD="$CMD --data-dir $DATA_DIR"

if [ -n "$REPO_ID" ]; then
    CMD="$CMD --repo-id $REPO_ID"
fi

if [ "$SPLIT" != "0.0" ]; then
    CMD="$CMD --split $SPLIT"
fi

if [ "$BINARY" = true ]; then
    CMD="$CMD --binary"
fi

if [ "$PRIVATE" = true ]; then
    CMD="$CMD --private"
fi

if [ -n "$TOKEN" ]; then
    CMD="$CMD --token $TOKEN"
fi

if [ -n "$MESSAGE" ]; then
    CMD="$CMD --commit-message \"$MESSAGE\""
fi

if [ "$CREATE_ONLY" = true ]; then
    CMD="$CMD --create-only"
fi

if [ -n "$METADATA_PATH" ]; then
    CMD="$CMD --metadata-path $METADATA_PATH"
fi

# Show configurations
echo ""
print_info "=== UPLOAD CONFIGURATION ==="
echo "Data directory: $DATA_DIR"
if [ -n "$REPO_ID" ]; then
    echo "Repository: $REPO_ID"
fi
echo "Train/test split: $SPLIT"
echo "Binary dataset: $BINARY"
echo "Private repository: $PRIVATE"
echo "Message: $MESSAGE"
echo "Local only: $CREATE_ONLY"
if [ -n "$METADATA_PATH" ]; then
    echo "Metadata: $METADATA_PATH"
fi
echo ""

# Confirm execution
if [ "$CREATE_ONLY" = false ]; then
    read -p "Do you want to continue with the upload? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Upload cancelled by user"
        exit 0
    fi
fi

# Execute upload
print_info "Starting process..."
echo "Command: $CMD"
echo ""

# Capture start time
START_TIME=$(date +%s)

# Execute command
if eval $CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "Process completed successfully!"
    print_info "Total time: ${DURATION}s"
    
    # Show information about results
    if [ "$CREATE_ONLY" = false ]; then
        echo ""
        print_info "=== UPLOAD INFORMATION ==="
        echo "Dataset available at: https://huggingface.co/datasets/$REPO_ID"
        echo "Use the dataset in your Python projects:"
        echo "   from datasets import load_dataset"
        echo "   dataset = load_dataset('$REPO_ID')"
    else
        echo ""
        print_info "=== LOCAL DATASET INFORMATION ==="
        echo "Dataset created in: $DATA_DIR"
        echo "Metadata saved in: $DATA_DIR/galaxy_metadata.pth"
    fi
    
else
    print_error "Error during process"
    exit 1
fi

print_success "Script completed!"
