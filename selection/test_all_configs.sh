#!/bin/bash
# WEAVER Configuration Testing Script
# Tests all configurations and saves outputs to log files

set -e  # Exit on any error

# Create logs directory
LOGS_DIR="test_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"

echo "🧪 Starting WEAVER configuration tests..."
echo "📁 Logs will be saved to: $LOGS_DIR"

# Function to run a config and capture output
run_config() {
    local config_path="$1"
    local config_name="$2"
    local log_file="$LOGS_DIR/${config_name}.log"
    
    echo "⏳ Testing: $config_name"
    echo "📄 Config: $config_path"
    echo "📝 Log: $log_file"
    
    # Record start time
    echo "=== WEAVER Test: $config_name ===" > "$log_file"
    echo "Started: $(date)" >> "$log_file"
    echo "Config: $config_path" >> "$log_file"
    echo "========================================" >> "$log_file"
    echo "" >> "$log_file"
    
    # Run the test and capture all output
    if python run.py --config-path="$config_path" --config-name="$config_name" logging=none >> "$log_file" 2>&1; then
        echo "✅ SUCCESS: $config_name"
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Completed: $(date)" >> "$log_file"
        echo "Status: SUCCESS" >> "$log_file"
    else
        echo "❌ FAILED: $config_name"
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Completed: $(date)" >> "$log_file"
        echo "Status: FAILED" >> "$log_file"
    fi
    echo ""
}

# Change to selection directory
cd "$(dirname "$0")"
if [ ! -f "run.py" ]; then
    echo "❌ Error: run.py not found. Make sure you're in the selection/ directory."
    exit 1
fi

echo "📍 Working directory: $(pwd)"
echo ""

# Test main configs (in configs/ directory)
echo "🔵 Testing main configurations..."
run_config "configs" "first_sample"
run_config "configs" "majority_vote" 
run_config "configs" "supervised"
run_config "configs" "weak_supervision"
run_config "configs" "weak_supervision_clustering"

# Test best configs (in configs/best_configs/ directory)
echo "🟢 Testing best configurations..."
run_config "configs/best_configs" "MATH-500_70B"
run_config "configs/best_configs" "MATH-500_8B"
run_config "configs/best_configs" "MMLU_70B"
run_config "configs/best_configs" "MMLU_8B"
run_config "configs/best_configs" "MMLU-Pro_70B"
run_config "configs/best_configs" "MMLU-Pro_8B"
run_config "configs/best_configs" "GPQA_70B"
run_config "configs/best_configs" "GPQA_8B"
run_config "configs/best_configs" "GPQA_1K"

echo "🎉 All tests completed!"
echo ""
echo "📊 Results summary:"
echo "==================="

# Count successes and failures
success_count=0
failure_count=0
total_count=0

for log_file in "$LOGS_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        total_count=$((total_count + 1))
        config_name=$(basename "$log_file" .log)
        
        if grep -q "Status: SUCCESS" "$log_file"; then
            echo "✅ $config_name"
            success_count=$((success_count + 1))
        elif grep -q "Status: FAILED" "$log_file"; then
            echo "❌ $config_name"
            failure_count=$((failure_count + 1))
        else
            echo "⚠️  $config_name (incomplete)"
            failure_count=$((failure_count + 1))
        fi
    fi
done

echo ""
echo "📈 Final Results:"
echo "  ✅ Successful: $success_count"
echo "  ❌ Failed: $failure_count"
echo "  📁 Total: $total_count"
echo ""
echo "📁 Detailed logs available in: $LOGS_DIR"

# Create a summary file
summary_file="$LOGS_DIR/SUMMARY.txt"
echo "WEAVER Configuration Test Summary" > "$summary_file"
echo "Generated: $(date)" >> "$summary_file"
echo "======================================" >> "$summary_file"
echo "" >> "$summary_file"
echo "Results:" >> "$summary_file"
echo "  Successful: $success_count" >> "$summary_file"
echo "  Failed: $failure_count" >> "$summary_file"
echo "  Total: $total_count" >> "$summary_file"
echo "" >> "$summary_file"

# Add individual results to summary
echo "Individual Results:" >> "$summary_file"
for log_file in "$LOGS_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        config_name=$(basename "$log_file" .log)
        if grep -q "Status: SUCCESS" "$log_file"; then
            echo "  ✅ $config_name" >> "$summary_file"
        else
            echo "  ❌ $config_name" >> "$summary_file"
        fi
    fi
done

echo "📄 Summary saved to: $summary_file"

if [ $failure_count -eq 0 ]; then
    echo ""
    echo "🎉 All configurations passed! WEAVER is ready for release."
    exit 0
else
    echo ""
    echo "⚠️  Some configurations failed. Check the logs for details."
    exit 1
fi