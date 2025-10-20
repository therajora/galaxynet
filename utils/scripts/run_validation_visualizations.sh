#!/bin/bash

# Script to execute validation visualizations
# Activates conda environment and runs visualizations

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
    echo "Validation Visualizations - Galaxy Classification SDSS/S-PLUS"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --results FILE        Path to JSON results file (required)"
    echo "  --output-dir DIR      Output directory (default: validation/visualization)"
    echo "  --skip-basic          Skip basic plots generation"
    echo "  --skip-comparisons    Skip comparative plots generation"
    echo "  --skip-analysis       Skip detailed analysis generation"
    echo "  --help, -h           Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --results validation/tests/results/test_results.json"
    echo "  $0 --results results.json --output-dir custom/plots"
    echo "  $0 --results results.json --skip-analysis"
    echo ""
    echo "DESCRIPTION:"
    echo "  Generates complete visualizations of model validation results."
    echo ""
    echo "  Visualizations include:"
    echo "  • Basic plots (accuracy, metrics, families)"
    echo "  • Comparative plots (ranking, complexity, correlation)"
    echo "  • Detailed analysis (best model, errors, evolution)"
    echo "  • Final report with insights"
}

# Check if in correct directory
check_project_root() {
    if [ ! -f "README.md" ] || [ ! -d "validation" ]; then
        print_error "Execute this script from the project root directory!"
        print_info "Current directory: $(pwd)"
        print_info "Expected: directory with README.md and validation/ folder"
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
    dependencies=("matplotlib" "seaborn" "pandas" "numpy" "sklearn")
    
    for dep in "${dependencies[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            print_warning "Dependency '$dep' not found. Installing..."
            pip install "$dep"
        fi
    done
    
    print_success "All dependencies are available!"
}

# Check if results file exists
check_results_file() {
    if [ ! -f "$1" ]; then
        print_error "Results file not found: $1"
        print_info "Execute validation tests first:"
        print_info "  ./utils/scripts/run.sh validation-tests"
        exit 1
    fi
}

# Main function
main() {
    echo "VALIDATION VISUALIZATIONS - GALAXY CLASSIFICATION SDSS/S-PLUS"
    echo "=================================================================="
    
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
    python_cmd="python validation/visualization/run_visualizations.py"
    
    # Add arguments
    results_file=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --results)
                results_file="$2"
                python_cmd="$python_cmd --results $2"
                shift 2
                ;;
            --output-dir)
                python_cmd="$python_cmd --output-dir $2"
                shift 2
                ;;
            --skip-basic)
                python_cmd="$python_cmd --skip-basic"
                shift
                ;;
            --skip-comparisons)
                python_cmd="$python_cmd --skip-comparisons"
                shift
                ;;
            --skip-analysis)
                python_cmd="$python_cmd --skip-analysis"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check if results file was provided
    if [ -z "$results_file" ]; then
        print_error "Results file is required!"
        print_info "Use: $0 --results validation/tests/results/test_results.json"
        exit 1
    fi
    
    # Check if file exists
    check_results_file "$results_file"
    
    # Execute visualizations
    print_info "Executing validation visualizations..."
    print_info "Command: $python_cmd"
    echo ""
    
    if eval $python_cmd; then
        print_success "Validation visualizations executed successfully!"
        echo ""
        print_info "Visualizations saved in: validation/visualization/"
        print_info "Report: validation/visualization/visualization_report.md"
        print_info "Basic plots: validation/visualization/plots/"
        print_info "Comparative plots: validation/visualization/comparisons/"
        print_info "Detailed analysis: validation/visualization/analysis/"
    else
        print_error "Error during visualization execution!"
        exit 1
    fi
    
    # Deactivate conda environment
    conda deactivate
    
    echo ""
    print_success "Process completed successfully!"
}

# Execute main function with all arguments
main "$@"
