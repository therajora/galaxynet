#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No color

# Default variables
MODE=""
MODEL_NAME=""
MODEL_PATH=""
SDSS_DATA_DIR="data/complete_sdss"
SPLUS_DATA_DIR="data/complete_splus"
OUTPUT_DIR="domain_shift_results"
DEVICE="auto"
BATCH_SIZE=32
SEED=42
RESULTS_DIR="results/models"

# Function to display help
show_help() {
    echo -e "${BLUE}=== Domain Shift Analysis Script ===${NC}"
    echo -e "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -m, --mode MODE         Operation mode: single, multiple, find"
    echo "  -n, --model-name NAME   Model name (for single mode)"
    echo "  -p, --model-path PATH   Path to trained model .pth file"
    echo "  -s, --sdss-data-dir DIR SDSS data directory (default: ${SDSS_DATA_DIR})"
    echo "  -l, --splus-data-dir DIR S-PLUS data directory (default: ${SPLUS_DATA_DIR})"
    echo "  -o, --output-dir DIR    Directory to save results (default: ${OUTPUT_DIR})"
    echo "  --device DEVICE         Device to use (auto, cuda, cpu). (default: ${DEVICE})"
    echo "  --batch-size SIZE       Batch size (default: ${BATCH_SIZE})"
    echo "  --results-dir DIR       Directory where trained models are saved (default: ${RESULTS_DIR})"
    echo "  --seed NUM              Seed for reproducibility (default: ${SEED})"
    echo "  -h, --help              Show this help"
    echo ""
    echo "MODES:"
    echo "  single                  Analyze domain shift for a specific model"
    echo "  multiple                Analyze domain shift for all trained models"
    echo "  find                    Find available trained models"
    echo ""
    echo "EXAMPLES:"
    echo "  $(basename "$0") --mode find"
    echo "  $(basename "$0") --mode single -n resnet50_v1 -p results/models/galaxy_resnet50_20230101/best_model.pth"
    echo "  $(basename "$0") --mode multiple -s data/complete_sdss -l data/complete_splus"
    echo ""
    echo "DESCRIPTION:"
    echo "  This script analyzes the domain shift problem when models trained"
    echo "  on SDSS data are applied to S-PLUS data. Calculates metrics such as:"
    echo "  - Performance drop between domains"
    echo "  - Distances between feature distributions"
    echo "  - Shift in prediction confidence"
    echo "  - Calibration metrics"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -m|--mode) MODE="$2"; shift 2 ;;
        -n|--model-name) MODEL_NAME="$2"; shift 2 ;;
        -p|--model-path) MODEL_PATH="$2"; shift 2 ;;
        -s|--sdss-data-dir) SDSS_DATA_DIR="$2"; shift 2 ;;
        -l|--splus-data-dir) SPLUS_DATA_DIR="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo -e "${RED}Error: Unknown option '$1'${NC}"; show_help; exit 1 ;;
    esac
done

# Mode validation
if [ -z "$MODE" ]; then
    echo -e "${RED}Error: Operation mode (--mode) is required.${NC}"
    show_help
    exit 1
fi

# Path to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR" || { echo -e "${RED}Error: Could not change to project root directory.${NC}"; exit 1; }

echo -e "${BLUE}Executing domain shift analysis...${NC}"

# Activate conda environment (if not active)
if ! conda info --envs | grep -q "galaxynet"; then
    echo -e "${RED}Error: Conda environment 'galaxynet' not found. Please create or activate it manually.${NC}"
    exit 1
fi

if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "galaxynet" ]]; then
    echo -e "${BLUE}Activating conda environment galaxynet...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh" || { echo -e "${RED}Error: Could not load conda initialization script.${NC}"; exit 1; }
    conda activate galaxynet || { echo -e "${RED}Error: Could not activate environment 'galaxynet'.${NC}"; exit 1; }
    echo -e "${GREEN}Environment galaxynet activated${NC}"
fi

echo -e "${BLUE}=== ANALYSIS CONFIGURATION ===${NC}"
echo -e "Mode: ${MODE}"
if [ -n "$MODEL_NAME" ]; then echo -e "Model: ${MODEL_NAME}"; fi
if [ -n "$MODEL_PATH" ]; then echo -e "Path: ${MODEL_PATH}"; fi
echo -e "SDSS Data: ${SDSS_DATA_DIR}"
echo -e "S-PLUS Data: ${SPLUS_DATA_DIR}"
echo -e "Output: ${OUTPUT_DIR}"
echo -e "Device: ${DEVICE}"
echo -e "Batch Size: ${BATCH_SIZE}"
echo -e "${BLUE}=======================================${NC}"

# Check if data directories exist
if [ ! -d "$SDSS_DATA_DIR" ]; then
    echo -e "${RED}ERROR: SDSS directory not found: ${SDSS_DATA_DIR}${NC}"
    exit 1
fi

if [ ! -d "$SPLUS_DATA_DIR" ]; then
    echo -e "${RED}ERROR: S-PLUS directory not found: ${SPLUS_DATA_DIR}${NC}"
    exit 1
fi

read -p "$(echo -e "${YELLOW}Do you want to start the analysis with this configuration? (y/N): ${NC}")" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Analysis cancelled.${NC}"
    exit 0
fi

START_TIME=$(date +%s)

# Build Python command
PYTHON_CMD="python domain_shift/analyze_domain_shift.py"
PYTHON_CMD+=" --mode \"$MODE\""
if [ -n "$MODEL_NAME" ]; then PYTHON_CMD+=" --model-name \"$MODEL_NAME\""; fi
if [ -n "$MODEL_PATH" ]; then PYTHON_CMD+=" --model-path \"$MODEL_PATH\""; fi
PYTHON_CMD+=" --sdss-data-dir \"$SDSS_DATA_DIR\""
PYTHON_CMD+=" --splus-data-dir \"$SPLUS_DATA_DIR\""
PYTHON_CMD+=" --output-dir \"$OUTPUT_DIR\""
PYTHON_CMD+=" --device \"$DEVICE\""
PYTHON_CMD+=" --batch-size \"$BATCH_SIZE\""
PYTHON_CMD+=" --results-dir \"$RESULTS_DIR\""
PYTHON_CMD+=" --seed \"$SEED\""

echo -e "${BLUE}Starting domain shift analysis...${NC}"
echo -e "${BLUE}Command: ${PYTHON_CMD}${NC}"

# Execute Python script
if eval "$PYTHON_CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}Domain shift analysis completed successfully!${NC}"
    echo -e "${BLUE}Total time: ${DURATION}s${NC}"
    echo -e "${BLUE}=== RESULTS ===${NC}"
    echo -e "Results saved in: ${OUTPUT_DIR}"
    echo -e "Metrics: ${OUTPUT_DIR}/*/domain_shift_summary.json"
    echo -e "Plots: ${OUTPUT_DIR}/*/*.png"
    echo -e "Reports: ${OUTPUT_DIR}/*/domain_shift_results.json"
    
    # Show summary of generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${CYAN}Generated files:${NC}"
        find "$OUTPUT_DIR" -name "*.json" -o -name "*.png" | head -10 | while read -r file; do
            echo -e "  - ${file}"
        done
        total_files=$(find "$OUTPUT_DIR" -name "*.json" -o -name "*.png" | wc -l)
        if [ "$total_files" -gt 10 ]; then
            echo -e "  ... and $((total_files - 10)) more files"
        fi
    fi
    
    echo -e "${GREEN}Domain shift analysis finished!${NC}"
else
    echo -e "${RED}Error during domain shift analysis${NC}"
    exit 1
fi

echo -e "${GREEN}Script completed!${NC}"
