# Weaver Distillation

While Weaver significantly improves response selection accuracy by combining multiple weak verifiers, running these verifiers (8B-72B parameters each) on each candidate response can be computationally expensive. To address this challenge, we can **distill** Weaver into a compact  (400M parameter) cross-encoder model that captures the combined strengths of the full Weaver ensemble while requiring only a fraction of the compute. We find that our distilled model retains 98.7% of Weaver's accuracy while reducing verification compute by up to 99.97%! The cross-encoder takes concatenated query-response pairs as input and outputs the probability that the response is correct, following the same interface as Weaver's weighted ensemble decision.

For initial setup and dependencies, see [README.md](../README.md).

## Pre-trained Models

We provide several task-specific distilled models trained on Weaver pseudolabels:

- [Weaver_Distilled_for_MATH500](https://huggingface.co/hazyresearch/Weaver_Distilled_for_MATH500)
- [Weaver_Distilled_for_GPQA](https://huggingface.co/hazyresearch/Weaver_Distilled_for_GPQA) 
- [Weaver_Distilled_for_MMLU-Pro](https://huggingface.co/hazyresearch/Weaver_Distilled_for_MMLU-Pro)
- [Weaver_Distilled_All_Datasets_gte-Qwen2-1.5B-instruct](https://huggingface.co/hazyresearch/Weaver_Distilled_All_Datasets_gte-Qwen2-1.5B-instruct)
- [hazyresearch/Weaver_Distilled_All_Datasets_ModernBERT-large](https://huggingface.co/hazyresearch/Weaver_Distilled_All_Datasets_ModernBERT-large)

## Training Your Own Distilled Model (GPU Required)

### Training Script

Use `train.py` to train a distilled model, (dataset_path is an example):

```bash
python train.py \
    --dataset_path "hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1" \
    --score_columns weaver_scores \
    --model_name "answerdotai/ModernBERT-large" \
    --num_epochs 30 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --max_length 4096 \
    --output_dir "checkpoints/ModernBERT-large_FINAL_V1.1.4/MATH500_with_Llama_3.1_70B_Instruct_v1_NAIVE_ENSEMBLE_30EPOCHS" \
    --shuffle_samples \
    --use_naive_ensemble \
    --use_wandb
```

**Note**: Flash attention is disabled by default to avoid GLIBC version compatibility issues. If you have GLIBC > 2.32 and have installed flash-attn (`pip install flash-attn`), you can enable flash attention for better performance by uncommenting line 134 (`"use_flash_attention_2": True`) in `train.py`. You can also uncomment line 135 (`"attn_implementation": "eager"`) to use eager attention, or leave both commented to proceed with normal attention.

**Key Arguments:**
- `--dataset_path`: HuggingFace dataset with Weaver pseudolabels (local or remote)
- `--score_columns`: Column(s) containing Weaver aggregated scores obtained by running Weaver on the dataset
- `--model_name`: Base transformer model (we recommend ModernBERT-large)
- `--max_rows`: Limit training data size for faster iteration

### Dataset Format

Your training dataset should contain:
- `instruction`: The query/problem statement
- `samples`: List of candidate responses
- `weaver_scores`: Weaver's aggregated scores for each response

### Evaluation

Evaluate your trained model with `evaluate_cross_encoder.py`:

```bash
python evaluate.py \
  --model_name "answerdotai/ModernBERT-large" \
  --checkpoint_path "../checkpoints/selection__hazyresearch__GPQA_with_Llama_3.1_8B_Instruct_v1/20250606_140631.ckpt" \ # example path of your trained model
  --dataset_path "hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1" \
  --dataset_split "data" \
  --start_row 581 \
  --end_row 646 # evaluate on last 10% of the data
```

You can also evaluate our publicly released distilled models using variations of the following command:

```bash
python evaluate.py \
  --model_name "answerdotai/ModernBERT-large" \
  --checkpoint_path "hazyresearch/Weaver_Distilled_for_MMLU-Pro" \
  --dataset_path "hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1" \
  --dataset_split "data" \
  --start_row 450 \
  --end_row 500 # evaluate on last 10% of the data
```

The evaluation script reports:
- **Selection@1**: Accuracy when picking the highest-scored response
- **Top-K Average Score**: Accuracy when averaging scores for duplicate answers
- **Comparison with baselines**: Performance vs. majority voting and individual verifiers

## Performance Results

On GPQA Diamond with Llama 3.3 70B generations:

| Method | Accuracy | Compute Cost | Hardware |
|--------|----------|--------------|----------|
| Weaver (Full) | 66.4% | 35.35 ExaFLOPs | 8-GPU nodes × 30+ verifiers |
| Weaver (Distilled) | 65.3% | 1.01 ExaFLOPs | Single A100 |
| Majority Voting | 47.4% | ~0 ExaFLOPs | None |

## Tips for Best Results

1. **Model Selection**: ModernBERT-large provides the best accuracy/efficiency trade-off
2. **Data Quality**: Higher-quality Weaver pseudolabels lead to better distilled models
3. **Task-Specific Training**: Train separate models for different domains (math, science, etc.)
4. **Hyperparameter Tuning**: Adjust learning rate and batch size based on your dataset size

## Implementation Notes

- Cross-encoders use concatenated query-response pairs as input
- Training uses regression loss on Weaver's continuous scores
- The distilled model outputs probabilities that can be directly used for response selection
- Evaluation supports both binary classification metrics and selection accuracy

For more details on the distillation methodology, see Section 6 of our paper.
