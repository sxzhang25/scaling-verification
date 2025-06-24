#!/bin/bash

# test_generation.sh
# Main script to run all generation tests

set -e  # Exit on error

# Create test directories
mkdir -p tests/generation/results
mkdir -p tests/generation/datasets

# Function to run a test and validate results
run_test() {
    local test_name=$1
    local test_script=$2
    
    echo "Running test: $test_name"
    bash $test_script
    
    # Validate results using Python
    python tests/generation/test_utils.py validate $test_name
}

# Run tests for each dataset type
echo "Running MATH500 test..."
run_test "math" "tests/generation/test_math.sh"

echo "Running GPQA test..."
run_test "gpqa" "tests/generation/test_gpqa.sh"

echo "Running MMLU test..."
run_test "mmlu" "tests/generation/test_mmlu.sh"

echo "Running MMLU-Pro test..."
run_test "mmlu_pro" "tests/generation/test_mmlu_pro.sh"

# Run reproduction test
echo "Running reproduction test..."
run_test "reproduction" "tests/generation/test_reproduction.sh"

echo "All tests completed!"