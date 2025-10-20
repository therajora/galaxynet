#!/bin/bash

# Script for Execution of Validation Tests
# Executes tests with dummy data, generates visualizations and tables

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# Path to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}"
echo "=================================================================================="
echo "                    MODEL VALIDATION TESTS"
echo "                      Execution with Dummy Data"
echo "=================================================================================="
echo -e "${NC}"

# Function to activate conda environment
activate_conda() {
    echo -e "${BLUE}Activating conda environment 'galaxynet'...${NC}"
    
    # Initialize conda if necessary
    if ! command -v conda &> /dev/null; then
        echo -e "${YELLOW}Conda not found, trying to initialize...${NC}"
        eval "$(conda shell.bash hook)"
    fi
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate galaxynet
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Environment 'galaxynet' activated successfully${NC}"
    else
        echo -e "${RED}Error activating environment 'galaxynet'${NC}"
        echo -e "${YELLOW}Check if environment exists: conda env list${NC}"
        exit 1
    fi
}

# Function to display help
show_help() {
    echo -e "${YELLOW}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo -e "${GREEN}OPTIONS:${NC}"
    echo -e "  ${BLUE}--num-samples N${NC}        Number of samples for test (default: 1000)"
    echo -e "  ${BLUE}--output-dir DIR${NC}       Output directory (default: validation/tests/results)"
    echo -e "  ${BLUE}--skip-tests${NC}           Skip test execution"
    echo -e "  ${BLUE}--skip-visualizations${NC}  Skip visualization generation"
    echo -e "  ${BLUE}--skip-tables${NC}          Skip table generation"
    echo -e "  ${BLUE}--quick${NC}                Quick execution (500 samples, summary only)"
    echo -e "  ${BLUE}--full${NC}                 Full execution (2000 samples, all visualizations)"
    echo -e "  ${BLUE}--help${NC}                 Show this help"
    echo ""
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo -e "  ${BLUE}$0 --quick${NC}                    # Quick execution"
    echo -e "  ${BLUE}$0 --full${NC}                     # Full execution"
    echo -e "  ${BLUE}$0 --num-samples 500${NC}          # Test with 500 samples"
    echo -e "  ${BLUE}$0 --skip-tests --skip-visualizations${NC}  # Tables only"
    echo ""
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 not found${NC}"
        exit 1
    fi
    
    # Check if in correct directory
    if [ ! -f "$PROJECT_DIR/validation/tests/run_tests.py" ]; then
        echo -e "${RED}Test script not found${NC}"
        echo -e "${YELLOW}Execute this script from the project root directory${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Dependencies verified${NC}"
}

# Function to execute tests
run_tests() {
    local args="$1"
    
    echo -e "${BLUE}Executing validation tests...${NC}"
    echo -e "${YELLOW}Project directory: $PROJECT_DIR${NC}"
    echo ""
    
    cd "$PROJECT_DIR"
    
    # Execute Python script
    python3 validation/tests/run_tests.py $args
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Tests executed successfully!${NC}"
    else
        echo -e "${RED}Error executing tests${NC}"
        exit $exit_code
    fi
}

# Function to show results
show_results() {
    local output_dir="$1"
    
    echo -e "${BLUE}Generated results:${NC}"
    echo ""
    
    if [ -d "$output_dir" ]; then
        echo -e "${GREEN}Directory: $output_dir${NC}"
        
        # List generated files
        if [ -f "$output_dir/test_results.json" ]; then
            echo -e "  JSON Results: test_results.json"
        fi
        
        if [ -f "$output_dir/final_report.md" ]; then
            echo -e "  Final Report: final_report.md"
        fi
        
        if [ -d "$output_dir/visualizations" ]; then
            local viz_count=$(find "$output_dir/visualizations" -name "*.png" | wc -l)
            echo -e "  Visualizations: $viz_count charts"
        fi
        
        if [ -d "$output_dir/tables" ]; then
            local table_count=$(find "$output_dir/tables" -name "*.html" | wc -l)
            echo -e "  Tables: $table_count HTML tables"
        fi
        
        echo ""
        echo -e "${YELLOW}To view results:${NC}"
        echo -e "  ${BLUE}cat $output_dir/final_report.md${NC}"
        echo -e "  ${BLUE}ls -la $output_dir/visualizations/${NC}"
        echo -e "  ${BLUE}ls -la $output_dir/tables/${NC}"
    else
        echo -e "${RED}Results directory not found${NC}"
    fi
}

# Process arguments
NUM_SAMPLES=1000
OUTPUT_DIR="validation/tests/results"
SKIP_TESTS=""
SKIP_VIZ=""
SKIP_TABLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="--skip-tests"
            shift
            ;;
        --skip-visualizations)
            SKIP_VIZ="--skip-visualizations"
            shift
            ;;
        --skip-tables)
            SKIP_TABLES="--skip-tables"
            shift
            ;;
        --quick)
            NUM_SAMPLES=500
            SKIP_VIZ="--skip-visualizations"
            shift
            ;;
        --full)
            NUM_SAMPLES=2000
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

# Main execution
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Samples: $NUM_SAMPLES"
echo -e "  Directory: $OUTPUT_DIR"
echo -e "  Skip tests: ${SKIP_TESTS:+Yes}"
echo -e "  Skip visualizations: ${SKIP_VIZ:+Yes}"
echo -e "  Skip tables: ${SKIP_TABLES:+Yes}"
echo ""

# Activate conda environment
activate_conda

# Check dependencies
check_dependencies

# Execute tests
run_tests "--num-samples $NUM_SAMPLES --output-dir $OUTPUT_DIR $SKIP_TESTS $SKIP_VIZ $SKIP_TABLES"

# Show results
show_results "$PROJECT_DIR/$OUTPUT_DIR"

echo ""
echo -e "${GREEN}Execution completed!${NC}"
