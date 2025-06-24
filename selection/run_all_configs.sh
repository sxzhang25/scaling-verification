#!/bin/bash

# Run WEAVER scoring for all configs
echo "Running WEAVER scoring for all configs..."

python run.py --config-path="configs/best_configs" --config-name="GPQA_1K" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="GPQA_70B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="GPQA_8B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MATH-500_70B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MATH-500_8B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MMLU-Pro_70B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MMLU-Pro_8B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MMLU_8B" # data_cfg.save_weaver_scores=true
python run.py --config-path="configs/best_configs" --config-name="MMLU_70B" # data_cfg.save_weaver_scores=true

echo "All configs completed!"