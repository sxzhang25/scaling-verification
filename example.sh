# Example commands to run Weaver end to end

# Generation: Generate reasoning samples and collect verifier scores
cd generation

echo "Step 1: Generating reasoning samples..."
python generate_reasoning_samples.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --dataset mmlu \
    --output_path ../datasets/MMLU_with_Llama_3.1_70B_Instruct_samples.hf \
    --max_rows 10 \
    --samples_per_instruction 2 \
    --temperature 0.7

echo "Step 2: Evaluating generated samples for correctness..."
python unified_evaluation.py \
    --input_path ../datasets/MMLU_with_Llama_3.1_70B_Instruct_samples.hf \
    --output_path ../datasets/MMLU_with_Llama_3.1_70B_Instruct_evaluated.hf \
    --dataset_type mmlu

echo "Step 3: Scoring samples with reward models and LM judges..."
python unified_RMs_and_LM_Judges.py \
    --dataset_path ../datasets/MMLU_with_Llama_3.1_70B_Instruct_evaluated.hf \
    --output_path ../datasets/MMLU_with_Llama_3.1_70B_Instruct.hf \
    --push_to_hub <YOUR_HF_ORG>/MMLU_with_Llama_3.1_70B_Instruct \
    --reward_models GRM,ArmorRM,URM,QRM \
    --lm_judges Llama-3.1-8B-Instruct-Together,GPT-4o-mini \
    --batch_size 4

# Selection: Use Weaver to select the best candidate generations from your dataset
cd .. && cd selection

echo "Step 4: Running Weaver selection on generated dataset..."
python run.py \
    --config-path="configs" \
    --config-name="weak_supervision" \
    data_cfg.dataset_path="<YOUR_HF_ORG>/MMLU_with_Llama_3.1_70B_Instruct" \
    data_cfg.save_weaver_scores=true

# Distillation: Train a distilled cross encoder on the weaver scores in your dataset
cd .. && cd distillation

echo "Step 5: Training distilled cross-encoder model..."
python train.py \
    --dataset_path "<YOUR_HF_ORG>/MMLU_with_Llama_3.1_70B_Instruct" \
    --score_columns weaver_scores \
    --model_name answerdotai/ModernBERT-large \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir ../checkpoints \
    --use_wandb

echo "Step 6: Evaluating distilled model performance..."
python evaluate.py \
  --model_name "answerdotai/ModernBERT-large" \
  --checkpoint_path "distillation/checkpoints/<YOUR_HF_ORG>__MMLU_with_Llama_3.1_70B_Instruct" \
  --dataset_path "hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1" \
  --dataset_split "data" \
  --start_row 450 \
  --end_row 500 # evaluate on last 10% of the data