#!/usr/bin/env bash
set -euo pipefail

# =============================
# Run main.py with 3 dataset orders and 3 random seeds
# =============================

# Define three dataset permutation orders
orders=(
  "dbpedia amazon yahoo agnews"
  "dbpedia amazon agnews yahoo"
  "yahoo amazon agnews dbpedia"
)

# Define random seeds to test
seeds=(438 689 251)

# Base directory to save models
base_save_dir="./saved_models"

# Ensure save base exists
mkdir -p "$base_save_dir"

# Backup original main.py once
cp main.py main.py.orig
for order in "${orders[@]}"; do
  for seed in "${seeds[@]}"; do
    # Split order into array
    IFS=' ' read -r -a ds_arr <<< "$order"
    # Create Python list literal: "dbpedia","amazon",...
    python_list_literal=$(printf '"%s",' "${ds_arr[@]}" | sed 's/,$//')

    # Replace the datasets line in main.py
    # Assumes line starts with '    datasets = ['
    sed -i "s/^\s*datasets = \[.*\]/    datasets = [${python_list_literal}]/" main.py

    # Define output directory for this configuration
    # save_dir="${base_save_dir}/${order// /_}/seed_${seed}"
    # mkdir -p "$save_dir"

    echo "=============================================="
    echo "Running seed=$seed, order=$order"
    echo "Saving to $base_save_dir"
    echo "=============================================="

    # Execute the multi-task continual learning pipeline
    python main.py --seed "$seed" --save_dir "$base_save_dir"

    echo "Completed seed=$seed, order=$order"
    echo
  done
done

echo "All experiments finished. Restoring original main.py"
# Restore original main.py
mv main.py.orig main.py
