#!/bin/bash

# Script to execute data augmentation tests
# Galaxy Classification SDSS/S-PLUS

set -e  # Para em caso de erro

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to show help
show_help() {
    echo -e "${BLUE}DATA AUGMENTATION TESTS - GALAXY CLASSIFICATION SDSS/S-PLUS${NC}"
    echo "=================================================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --num-samples N        Number of samples per configuration (default: 5)"
    echo "  --num-augmentations N  Number of augmentations per sample (default: 8)"
    echo "  --output-dir DIR       Output directory for results"
    echo "  --visualizations-dir DIR Output directory for visualizations"
    echo "  --skip-tests           Skip test execution"
    echo "  --skip-visualizations  Skip visualization generation"
    echo "  --quick                Quick execution (3 samples, 4 augmentations)"
    echo "  --full                 Full execution (8 samples, 12 augmentations)"
    echo "  --help                 Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --quick                    # Quick execution"
    echo "  $0 --full                     # Full execution"
    echo "  $0 --num-samples 10           # 10 samples per configuration"
    echo "  $0 --skip-tests               # Visualizations only"
    echo ""
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking Python dependencies...${NC}"
    
    # List of required dependencies
    dependencies=("numpy" "matplotlib" "PIL" "albumentations")
    
    missing_deps=()
    for dep in "${dependencies[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        echo -e "${GREEN}All dependencies are available!${NC}"
    else
        echo -e "${RED}Missing dependencies: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}Install with: pip install ${missing_deps[*]}${NC}"
        exit 1
    fi
}

# Function to execute tests
run_augment_tests() {
    echo -e "${BLUE}Executing data augmentation tests...${NC}"
    
    # Build Python command
    python_cmd="python augment/tests/run_tests.py"
    
    # Add arguments
    if [ "$NUM_SAMPLES" != "" ]; then
        python_cmd="$python_cmd --num-samples $NUM_SAMPLES"
    fi
    
    if [ "$NUM_AUGMENTATIONS" != "" ]; then
        python_cmd="$python_cmd --num-augmentations $NUM_AUGMENTATIONS"
    fi
    
    if [ "$OUTPUT_DIR" != "" ]; then
        python_cmd="$python_cmd --output-dir $OUTPUT_DIR"
    fi
    
    if [ "$VISUALIZATIONS_DIR" != "" ]; then
        python_cmd="$python_cmd --visualizations-dir $VISUALIZATIONS_DIR"
    fi
    
    if [ "$SKIP_TESTS" = true ]; then
        python_cmd="$python_cmd --skip-tests"
    fi
    
    if [ "$SKIP_VISUALIZATIONS" = true ]; then
        python_cmd="$python_cmd --skip-visualizations"
    fi
    
    echo -e "${BLUE}Command: $python_cmd${NC}"
    echo ""
    
    # Execute command
    eval $python_cmd
}

# Main function
main() {
    echo -e "${BLUE}DATA AUGMENTATION TESTS - GALAXY CLASSIFICATION SDSS/S-PLUS${NC}"
    echo "=================================================================="
    
    # Check if in correct directory
    if [ ! -f "augment/tests/run_tests.py" ]; then
        echo -e "${RED}Execute this script from the project root directory${NC}"
        echo -e "${YELLOW}Current directory: $(pwd)${NC}"
        exit 1
    fi
    
    # Activate conda environment
    echo -e "${BLUE}Activating conda environment 'galaxynet'...${NC}"
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate galaxynet
        echo -e "${GREEN}Environment 'galaxynet' activated${NC}"
    else
        echo -e "${YELLOW}Conda not found, continuing without activation${NC}"
    fi
    
    # Check dependencies
    check_dependencies
    
    # Execute tests
    run_augment_tests
    
    echo -e "${GREEN}Augmentation tests executed successfully!${NC}"
    
    # Deactivate conda environment
    if command -v conda &> /dev/null; then
        conda deactivate
        echo -e "${BLUE}Conda environment deactivated${NC}"
    fi
    
    echo -e "${BLUE}Results saved in: augment/tests/results/${NC}"
    echo -e "${BLUE}Visualizations saved in: augment/tests/visualizations/${NC}"
    echo -e "${BLUE}Final report: augment/tests/results/final_report.md${NC}"
    
    echo -e "${GREEN}Process completed successfully!${NC}"
}

# Process arguments
NUM_SAMPLES=""
NUM_AUGMENTATIONS=""
OUTPUT_DIR=""
VISUALIZATIONS_DIR=""
SKIP_TESTS=false
SKIP_VISUALIZATIONS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-augmentations)
            NUM_AUGMENTATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --visualizations-dir)
            VISUALIZATIONS_DIR="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-visualizations)
            SKIP_VISUALIZATIONS=true
            shift
            ;;
        --quick)
            NUM_SAMPLES="3"
            NUM_AUGMENTATIONS="4"
            shift
            ;;
        --full)
            NUM_SAMPLES="8"
            NUM_AUGMENTATIONS="12"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW}Use --help to see available options${NC}"
            exit 1
            ;;
    esac
done

# Execute main function
main
