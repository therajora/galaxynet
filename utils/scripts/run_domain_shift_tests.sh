#!/bin/bash

# Script to execute domain shift tests
# Activates conda environment and runs tests with dummy data

set -e  # Para em caso de erro

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
    echo "Domain Shift Tests - Galaxy Classification SDSS/S-PLUS"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --quick              Execute quick test (500 samples per model)"
    echo "  --full               Execute full test (2000 samples per model)"
    echo "  --num-samples N      Number of samples per model (default: 1000)"
    echo "  --output-dir DIR     Output directory (default: domain_shift/tests/results)"
    echo "  --skip-visualizations  Skip visualization generation"
    echo "  --skip-tables        Skip table generation"
    echo "  --help, -h           Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --quick                    # Quick test"
    echo "  $0 --full                     # Full test"
    echo "  $0 --num-samples 1500         # Custom test"
    echo "  $0 --skip-visualizations      # Tables only"
    echo ""
    echo "DESCRIPTION:"
    echo "  Executes domain shift tests with dummy data to validate"
    echo "  the complete analysis pipeline between SDSS and S-PLUS."
    echo ""
    echo "  Tests include:"
    echo "  • Realistic dummy data generation"
    echo "  • Testing all available models"
    echo "  • Domain shift metrics calculation"
    echo "  • Comparative visualization generation"
    echo "  • Benchmark table creation"
    echo "  • Detailed final report"
}

# Check if in correct directory
check_project_root() {
    if [ ! -f "README.md" ] || [ ! -d "domain_shift" ]; then
        print_error "Execute this script from the project root directory!"
        print_info "Current directory: $(pwd)"
        print_info "Expected: directory with README.md and domain_shift/ folder"
        exit 1
    fi
}

# Check if conda environment is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found! Install Anaconda or Miniconda."
        exit 1
    fi
}

# Check if galaxynet environment exists
check_environment() {
    if ! conda env list | grep -q "galaxynet"; then
        print_warning "Environment 'galaxynet' not found!"
        print_info "Creating environment from conda_env/environment.yml..."
        
        if [ -f "conda_env/environment.yml" ]; then
            conda env create -f conda_env/environment.yml
            print_success "Environment 'galaxynet' created successfully!"
        else
            print_error "File conda_env/environment.yml not found!"
            print_info "Execute first: ./utils/scripts/run.sh setup-environment"
            exit 1
        fi
    fi
}

# Check Python dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    # List of required dependencies
    dependencies=("numpy" "matplotlib" "seaborn" "pandas" "tabulate")
    
    for dep in "${dependencies[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            print_warning "Dependency '$dep' not found. Installing..."
            pip install "$dep"
        fi
    done
    
    print_success "All dependencies are available!"
}

# Main function
main() {
    echo "DOMAIN SHIFT TESTS - GALAXY CLASSIFICATION SDSS/S-PLUS"
    echo "=============================================================="
    
    # Initial checks
    check_project_root
    check_conda
    check_environment
    
    # Activate conda environment
    print_info "Activating conda environment 'galaxynet'..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate galaxynet
    
    # Check dependencies
    check_dependencies
    
    # Build Python command
    python_cmd="python domain_shift/tests/run_tests.py"
    
    # Add arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --quick)
                python_cmd="$python_cmd --quick"
                shift
                ;;
            --full)
                python_cmd="$python_cmd --full"
                shift
                ;;
            --num-samples)
                python_cmd="$python_cmd --num-samples $2"
                shift 2
                ;;
            --output-dir)
                python_cmd="$python_cmd --output-dir $2"
                shift 2
                ;;
            --skip-visualizations)
                python_cmd="$python_cmd --skip-visualizations"
                shift
                ;;
            --skip-tables)
                python_cmd="$python_cmd --skip-tables"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Execute tests
    print_info "Executing domain shift tests..."
    print_info "Command: $python_cmd"
    echo ""
    
    if eval $python_cmd; then
        print_success "Domain shift tests executed successfully!"
        echo ""
        print_info "Results saved in: domain_shift/tests/results/"
        print_info "Final report: domain_shift/tests/results/domain_shift_final_report.md"
        print_info "Tables: domain_shift/tests/results/tables/"
        print_info "Visualizations: domain_shift/tests/results/visualizations/"
    else
        print_error "Error during test execution!"
        exit 1
    fi
    
    # Deactivate conda environment
    conda deactivate
    
    echo ""
    print_success "Process completed successfully!"
}

# Execute main function with all arguments
main "$@"
