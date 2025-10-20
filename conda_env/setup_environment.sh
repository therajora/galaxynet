#!/bin/bash

# Script to configure the galaxynet conda environment
# This script recreates the conda environment used in the Galaxy Classification project

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # Sem cor

echo -e "${BLUE}Configuring galaxynet conda environment...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "galaxynet"; then
    echo -e "${YELLOW}Environment 'galaxynet' already exists.${NC}"
    read -p "Do you want to recreate the environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n galaxynet -y
    else
        echo -e "${BLUE}Activating existing environment...${NC}"
        conda activate galaxynet
        echo -e "${GREEN}Environment activated!${NC}"
        exit 0
    fi
fi

# Create environment from environment.yml file
echo -e "${BLUE}Creating environment from environment.yml...${NC}"
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Environment created successfully!${NC}"
    else
        echo -e "${RED}Error creating environment from environment.yml${NC}"
        echo -e "${YELLOW}Trying to create environment manually...${NC}"
        
        # Create environment manually with Python 3.11
        conda create -n galaxynet python=3.11 -y
        conda activate galaxynet
        
        # Install main conda packages
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        conda install numpy pandas matplotlib seaborn scikit-learn -y
        conda install jupyter notebook -y
        
        # Install pip packages
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        fi
        
        echo -e "${GREEN}Environment created manually!${NC}"
    fi
else
    echo -e "${RED}File environment.yml not found!${NC}"
    exit 1
fi

# Activate environment
echo -e "${BLUE}Activating environment...${NC}"
conda activate galaxynet

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" == "galaxynet" ]]; then
    echo -e "${GREEN}Environment 'galaxynet' activated successfully!${NC}"
    echo -e "${BLUE}Python version: $(python --version)${NC}"
    echo -e "${BLUE}PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')${NC}"
else
    echo -e "${RED}Error activating environment${NC}"
    exit 1
fi

echo -e "${GREEN}Environment configuration completed!${NC}"
echo -e "${YELLOW}To activate the environment in the future, use: conda activate galaxynet${NC}"
