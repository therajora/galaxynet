#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# Default variables
MODE=""
MODEL_NAME=""
MODEL_PATH=""
DATA_DIR="data/complete_sdss"
OUTPUT_DIR=""
DEVICE="auto"
RESULTS_DIR="results/models"

# Function to display help
show_help() {
    echo -e "${BLUE}=== Model Validation Script ===${NC}"
    echo -e "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -m, --mode MODE         Operation mode: single, benchmark, find"
    echo "  -n, --model-name NAME   Model name (for single mode)"
    echo "  -p, --model-path PATH   Model path (for single mode)"
    echo "  -d, --data-dir DIR      Data directory (default: data/complete_sdss)"
    echo "  -o, --output-dir DIR    Output directory"
    echo "  --device DEVICE         Device: auto, cuda, cpu (default: auto)"
    echo "  --results-dir DIR       Results directory (default: results/models)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "MODES:"
    echo "  single                  Validate a single model"
    echo "  benchmark               Run benchmark of all trained models"
    echo "  find                    Find available trained models"
    echo ""
    echo "EXAMPLES:"
    echo "  $(basename "$0") --mode find"
    echo "  $(basename "$0") --mode single -n resnet50 -p results/models/galaxy_resnet50/best_model.pth"
    echo "  $(basename "$0") --mode benchmark -d data/complete_sdss"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -m|--mode) MODE="$2"; shift 2 ;;
        -n|--model-name) MODEL_NAME="$2"; shift 2 ;;
        -p|--model-path) MODEL_PATH="$2"; shift 2 ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo -e "${RED}Error: Unknown option '$1'${NC}"; show_help; exit 1 ;;
    esac
done

# Mode validation
if [ -z "$MODE" ]; then
    echo -e "${RED}Error: Mode (--mode) is required.${NC}"
    show_help
    exit 1
fi

if [[ "$MODE" != "single" && "$MODE" != "benchmark" && "$MODE" != "find" ]]; then
    echo -e "${RED}Error: Invalid mode '$MODE'. Use: single, benchmark, find${NC}"
    show_help
    exit 1
fi

# Specific validation for single mode
if [ "$MODE" = "single" ]; then
    if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error: For 'single' mode, --model-name and --model-path are required.${NC}"
        show_help
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
        exit 1
    fi
fi

# Path to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR" || { echo -e "${RED}Error: Could not change to project root directory.${NC}"; exit 1; }

echo -e "${BLUE}Executing model validation...${NC}"

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

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}=== VALIDATION CONFIGURATION ===${NC}"
echo -e "Mode: ${MODE}"
if [ "$MODE" = "single" ]; then
    echo -e "Model: ${MODEL_NAME}"
    echo -e "Path: ${MODEL_PATH}"
fi
echo -e "Data: ${DATA_DIR}"
echo -e "Device: ${DEVICE}"
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "Output: ${OUTPUT_DIR}"
fi
echo -e "${BLUE}=======================================${NC}"

# Build Python command
PYTHON_CMD="python validation/validate_models.py"
PYTHON_CMD+=" --mode \"$MODE\""
PYTHON_CMD+=" --data-dir \"$DATA_DIR\""
PYTHON_CMD+=" --device \"$DEVICE\""
PYTHON_CMD+=" --results-dir \"$RESULTS_DIR\""

if [ "$MODE" = "single" ]; then
    PYTHON_CMD+=" --model-name \"$MODEL_NAME\""
    PYTHON_CMD+=" --model-path \"$MODEL_PATH\""
fi

if [ -n "$OUTPUT_DIR" ]; then
    PYTHON_CMD+=" --output-dir \"$OUTPUT_DIR\""
fi

echo -e "${BLUE}Starting validation...${NC}"
echo -e "${BLUE}Command: ${PYTHON_CMD}${NC}"

START_TIME=$(date +%s)

# Execute Python script
if eval "$PYTHON_CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}Validation completed successfully!${NC}"
    echo -e "${BLUE}Total time: ${DURATION}s${NC}"
    
    # Show information about results
    if [ "$MODE" = "single" ]; then
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="validation_results/$MODEL_NAME"
        fi
        echo -e "${BLUE}=== RESULTS ===${NC}"
        echo -e "Results saved in: ${OUTPUT_DIR}"
        echo -e "Metrics: ${OUTPUT_DIR}/metrics.json"
        echo -e "Plots: ${OUTPUT_DIR}/*.png"
    elif [ "$MODE" = "benchmark" ]; then
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="benchmark_results"
        fi
        echo -e "${BLUE}=== BENCHMARK RESULTS ===${NC}"
        echo -e "Results saved in: ${OUTPUT_DIR}"
        echo -e "Comparison: ${OUTPUT_DIR}/benchmark_comparison.csv"
        echo -e "Plots: ${OUTPUT_DIR}/*.png"
    fi
    
else
    echo -e "${RED}Error during validation${NC}"
    exit 1
fi

echo -e "${GREEN}Script completed!${NC}"
