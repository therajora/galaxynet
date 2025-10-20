#!/bin/bash

# Central Script for Execution of All Project Scripts
# Maps and executes scripts from utils/scripts/ in an organized way

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No color

# Path to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Function to display banner
show_banner() {
    echo -e "${BLUE}"
    echo "=================================================================================="
    echo "                    GALAXY CLASSIFICATION - CENTRAL SCRIPT"
    echo "                           Unified Execution System"
    echo "=================================================================================="
    echo -e "${NC}"
}

# Function to display main help
show_help() {
    show_banner
    echo -e "${CYAN}Usage: $(basename "$0") [COMMAND] [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}AVAILABLE COMMANDS:${NC}"
    echo ""
    
    echo -e "${GREEN}MODEL TRAINING:${NC}"
    echo -e "  ${BLUE}train-pretrained${NC}     - Train pre-trained models (ResNet, EfficientNet, ViT)"
    echo -e "  ${BLUE}train-from-scratch${NC}   - Train CNN models from scratch"
    echo -e "  ${BLUE}generate-kaggle-configs${NC} - Generate optimized configurations for Kaggle"
    echo ""
    
    echo -e "${GREEN}VALIDATION AND BENCHMARK:${NC}"
    echo -e "  ${BLUE}validate${NC}            - Validate trained models"
    echo -e "  ${BLUE}benchmark${NC}           - Run benchmark of multiple models"
    echo -e "  ${BLUE}domain-shift${NC}        - Analyze domain shift between SDSS and S-PLUS"
    echo ""
    
    echo -e "${GREEN}DATA MANAGEMENT:${NC}"
    echo -e "  ${BLUE}augment${NC}             - Apply data augmentation to data"
    echo -e "  ${BLUE}upload-dataset${NC}      - Upload dataset to Hugging Face Hub"
    echo -e "  ${BLUE}clean-augmented${NC}     - Clean augmented data"
    echo -e "  ${BLUE}create-mosaics${NC}      - Generate visualization mosaics"
    echo ""
    
    echo -e "${GREEN}TESTS AND VALIDATION:${NC}"
    echo -e "  ${BLUE}validation-tests${NC}    - Run validation tests with dummy data"
    echo -e "  ${BLUE}domain-shift-tests${NC}  - Run domain shift tests with dummy data"
    echo -e "  ${BLUE}validation-plots${NC}    - Generate validation results visualizations"
    echo -e "  ${BLUE}augment-tests${NC}       - Run data augmentation tests"
    echo ""
    
    echo -e "${GREEN}INFORMATION:${NC}"
    echo -e "  ${BLUE}list${NC}                - List all available scripts"
    echo -e "  ${BLUE}help${NC}                - Show this help"
    echo ""
    
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo -e "  ${CYAN}$(basename "$0") train-pretrained -m efficientnet_b0${NC}"
    echo -e "  ${CYAN}$(basename "$0") generate-kaggle-configs --model all${NC}"
    echo -e "  ${CYAN}$(basename "$0") validate --mode benchmark${NC}"
    echo -e "  ${CYAN}$(basename "$0") domain-shift --mode multiple${NC}"
    echo -e "  ${CYAN}$(basename "$0") augment --input-dir data/raw --output-dir data/augmented${NC}"
    echo -e "  ${CYAN}$(basename "$0") upload-dataset --repo-name my-galaxy-dataset${NC}"
    echo -e "  ${CYAN}$(basename "$0") create-mosaics --mode all${NC}"
    echo -e "  ${CYAN}$(basename "$0") validation-tests --quick${NC}"
    echo -e "  ${CYAN}$(basename "$0") domain-shift-tests --quick${NC}"
    echo -e "  ${CYAN}$(basename "$0") validation-plots --results validation/tests/results/test_results.json${NC}"
    echo -e "  ${CYAN}$(basename "$0") augment-tests --quick${NC}"
    echo ""
    
    echo -e "${PURPLE}For specific command help:${NC}"
    echo -e "  ${CYAN}$(basename "$0") [COMMAND] --help${NC}"
}

# Function to list all scripts
list_scripts() {
    show_banner
    echo -e "${YELLOW}AVAILABLE SCRIPTS IN utils/scripts/:${NC}"
    echo ""
    
    echo -e "${GREEN}TRAINING:${NC}"
    echo -e "  ${BLUE}train_pretrained.sh${NC}  - Pre-trained model training"
    echo -e "  ${BLUE}from_scratch.sh${NC}      - Training models from scratch"
    echo ""
    
    echo -e "${GREEN}VALIDATION:${NC}"
    echo -e "  ${BLUE}validate_models.sh${NC}   - Model validation and benchmark"
    echo -e "  ${BLUE}analyze_domain_shift.sh${NC} - Domain shift analysis"
    echo ""
    
    echo -e "${GREEN}DATA:${NC}"
    echo -e "  ${BLUE}run_augment.sh${NC}       - Data augmentation"
    echo -e "  ${BLUE}upload_dataset.sh${NC}    - Upload to Hugging Face"
    echo -e "  ${BLUE}clean_augmented_data.sh${NC} - Clean augmented data"
    echo ""
    
    echo -e "${YELLOW}TIP: Use '$(basename "$0") [COMMAND] --help' for specific help${NC}"
}

# Function to execute pre-trained training script
run_train_pretrained() {
    echo -e "${BLUE}Executing pre-trained model training...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/train_pretrained.sh" "$@"
}

# Function to execute from-scratch training script
run_train_from_scratch() {
    echo -e "${BLUE}Executing from-scratch model training...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/from_scratch.sh" "$@"
}

# Function to execute validation script
run_validate() {
    echo -e "${BLUE}Executing model validation...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/validate_models.sh" "$@"
}

# Function to execute benchmark script
run_benchmark() {
    echo -e "${BLUE}Executing model benchmark...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/validate_models.sh" --mode benchmark "$@"
}

# Function to execute augmentation script
run_augment() {
    echo -e "${BLUE}Executing data augmentation...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/run_augment.sh" "$@"
}

# Function to execute upload script
run_upload_dataset() {
    echo -e "${BLUE}Executing dataset upload...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/upload_dataset.sh" "$@"
}

# Function to execute cleanup script
run_clean_augmented() {
    echo -e "${BLUE}Executing augmented data cleanup...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/clean_augmented_data.sh" "$@"
}

# Function to execute mosaic creation script
run_create_mosaics() {
    echo -e "${BLUE}Executing mosaic creation...${NC}"
    shift # Remove first argument (command)
    cd "$PROJECT_DIR"
    python data/visualization/create_mosaics.py "$@"
}

# Function to execute validation tests
run_validation_tests() {
    echo -e "${BLUE}Executing validation tests...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/run_validation_tests.sh" "$@"
}

# Function to execute domain shift tests
run_domain_shift_tests() {
    echo -e "${BLUE}Executing domain shift tests...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/run_domain_shift_tests.sh" "$@"
}

# Function to execute validation visualizations
run_validation_plots() {
    echo -e "${BLUE}Executing validation visualizations...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/run_validation_visualizations.sh" "$@"
}

run_augment_tests() {
    echo -e "${BLUE}Executing data augmentation tests...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/run_augment_tests.sh" "$@"
}

# Function to generate Kaggle configurations
run_generate_kaggle_configs() {
    echo -e "${BLUE}Generating configurations for Kaggle...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/generate_kaggle_configs.sh" "$@"
}

# Function to execute domain shift script
run_domain_shift() {
    echo -e "${BLUE}Executing domain shift analysis...${NC}"
    shift # Remove first argument (command)
    "$SCRIPT_DIR/analyze_domain_shift.sh" "$@"
}

# Function to show project status
show_status() {
    show_banner
    echo -e "${YELLOW}PROJECT STATUS:${NC}"
    echo ""
    
    # Check directory structure
    echo -e "${GREEN}Directory Structure:${NC}"
    if [ -d "$PROJECT_DIR/data" ]; then
        echo -e "  OK data/ - $(find "$PROJECT_DIR/data" -type f | wc -l) files"
    else
        echo -e "  ERROR data/ - Not found"
    fi
    
    if [ -d "$PROJECT_DIR/results" ]; then
        echo -e "  OK results/ - $(find "$PROJECT_DIR/results" -type f | wc -l) files"
    else
        echo -e "  ERROR results/ - Not found"
    fi
    
    if [ -d "$PROJECT_DIR/model" ]; then
        echo -e "  OK model/ - Model modules available"
    else
        echo -e "  ERROR model/ - Not found"
    fi
    
    echo ""
    
    # Check trained models
    echo -e "${GREEN}Trained Models:${NC}"
    if [ -d "$PROJECT_DIR/results/models" ]; then
        model_count=$(find "$PROJECT_DIR/results/models" -name "best_model.pth" | wc -l)
        if [ $model_count -gt 0 ]; then
            echo -e "  OK $model_count models found"
            find "$PROJECT_DIR/results/models" -name "best_model.pth" | while read model; do
                model_name=$(basename $(dirname "$model"))
                echo -e "    - $model_name"
            done
        else
            echo -e "  WARNING No trained models found"
        fi
    else
        echo -e "  ERROR Model directory not found"
    fi
    
    echo ""
    
    # Check conda environment
    echo -e "${GREEN}Conda Environment:${NC}"
    if [[ "$CONDA_DEFAULT_ENV" == "galaxynet" ]]; then
        echo -e "  OK Environment 'galaxynet' active"
    else
        echo -e "  WARNING Environment 'galaxynet' not active"
        echo -e "     Execute: conda activate galaxynet"
    fi
    
    echo ""
    echo -e "${PURPLE}Use '$(basename "$0") help' to see all available commands${NC}"
}

# Main function
main() {
    # Change to project directory
    cd "$PROJECT_DIR" || { echo -e "${RED}Error: Could not change to project directory.${NC}"; exit 1; }
    
    # Check if at least one argument was provided
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Extract command
    COMMAND="$1"
    
    # Execute appropriate command
    case "$COMMAND" in
        "train-pretrained"|"train_pretrained")
            run_train_pretrained "$@"
            ;;
        "train-from-scratch"|"train_from_scratch"|"from-scratch"|"from_scratch")
            run_train_from_scratch "$@"
            ;;
        "validate"|"validation")
            run_validate "$@"
            ;;
        "benchmark")
            run_benchmark "$@"
            ;;
        "augment"|"augmentation")
            run_augment "$@"
            ;;
        "upload-dataset"|"upload_dataset"|"upload")
            run_upload_dataset "$@"
            ;;
        "clean-augmented"|"clean_augmented"|"clean")
            run_clean_augmented "$@"
            ;;
        "create-mosaics"|"create_mosaics"|"mosaics")
            run_create_mosaics "$@"
            ;;
        "validation-tests"|"validation_tests"|"tests")
            run_validation_tests "$@"
            ;;
        "domain-shift-tests"|"domain_shift_tests"|"domain_tests")
            run_domain_shift_tests "$@"
            ;;
        "validation-plots"|"validation_plots"|"plots")
            run_validation_plots "$@"
            ;;
        "augment-tests"|"augment_tests"|"augment-tests")
            run_augment_tests "$@"
            ;;
        "generate-kaggle-configs"|"kaggle-configs"|"kaggle-config")
            run_generate_kaggle_configs "$@"
            ;;
        "domain-shift"|"domain_shift"|"domain")
            run_domain_shift "$@"
            ;;
        "list"|"scripts")
            list_scripts
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: '$COMMAND'${NC}"
            echo ""
            echo -e "${YELLOW}Available commands:${NC}"
            echo -e "  train-pretrained, train-from-scratch, generate-kaggle-configs, validate, benchmark"
            echo -e "  domain-shift, augment, upload-dataset, clean-augmented, create-mosaics, validation-tests, domain-shift-tests, validation-plots, augment-tests, list, status, help"
            echo ""
            echo -e "${PURPLE}Use '$(basename "$0") help' for complete help${NC}"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
