#!/bin/bash

# Script to clean old augmented data

echo "=== Cleaning Old Augmented Data ==="

# Remove old augmented data folders
echo "Removing old folders..."
rm -rf complete_sdss
rm -rf complete_splus
rm -rf augmented_sdss
rm -rf augmented_splus
rm -rf balanced_sdss
rm -rf balanced_splus

# Remove folders inside data/
echo "Removing folders inside data/..."
rm -rf data/complete_sdss
rm -rf data/complete_splus
rm -rf data/augmented_sdss
rm -rf data/augmented_splus
rm -rf data/balanced_sdss
rm -rf data/balanced_splus

# Remove temporary directories
echo "Removing temporary directories..."
rm -rf data/complete_sdss_temp
rm -rf data/complete_splus_temp
rm -rf data/augmented_sdss_temp
rm -rf data/augmented_splus_temp
rm -rf data/balanced_sdss_temp
rm -rf data/balanced_splus_temp

echo "Cleaning completed!"
echo "Now you can run ./run_augment.sh again"
