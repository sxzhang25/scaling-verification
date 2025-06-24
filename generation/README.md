# Generation

This directory contains scripts for generating the datasets used to evaluate reasoning verifiers as described in our paper. The pipeline creates datasets with model-generated reasoning samples paired with correctness labels from both ground truth evaluation and various reward models/LM judges.

For initial setup and dependencies, see [README.md](../README.md).

## Overview

Our dataset generation follows a **3-step** pipeline that produces the final evaluation datasets used in the paper. We generate multiple reasoning responses to benchmark questions using an LLM of choice, extract and verify answers against ground truth, then collect scores from various verifiers (reward models and LM judges). The resulting datasets enable evaluation of how well different verifiers (both individually and ensembled) can identify correct reasoning solutions to benchmark instructions.

## Released Datasets

You can access our released datasets on Hugging Face. For details on the dataset format, see [Final Dataset Format](#final-dataset-format).

- [MATH500 with Llama-3.1-8B-Instruct](https://huggingface.co/datasets/hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1)
- [MATH500 with Llama-3.1-70B-Instruct](https://huggingface.co/datasets/hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1)
- [GPQA with Llama-3.1-8B-Instruct](https://huggingface.co/datasets/hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1)
- [GPQA with Llama-3.1-70B-Instruct](https://huggingface.co/datasets/hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1)  
- [GPQA Diamond with Llama-3.1-70B-Instruct](https://huggingface.co/datasets/hazyresearch/GPQA_Diamond_with_Llama_3.1_70B_Instruct_up_to_1K_Samples_v1)
- [MMLU with Llama-3.1-8B-Instruct](https://huggingface.co/datasets/hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1)
- [MMLU with Llama-3.1-70B-Instruct](https://huggingface.co/datasets/hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1)
- [MMLU-Pro with Llama-3.1-8B-Instruct](https://huggingface.co/datasets/hazyresearch/MMLU-Pro_with_Llama_3.1_8B_Instruct_v1)
- [MMLU-Pro with Llama-3.1-70B-Instruct](https://huggingface.co/datasets/hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1)

[View all datasets →](https://huggingface.co/collections/hazyresearch/weaver-683798010b39c9653ddb9bd8)

```python
from datasets import load_dataset
dataset = load_dataset("hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1")["data"]
```

## Generate Your Own Datasets
First cd into the `generation` directory:
```bash
cd generation
```

### Step 1: Generate Samples
**Script**: [`generate_reasoning_samples.py`](generate_reasoning_samples.py)

Takes benchmark datasets and generates multiple reasoning responses for each problem using an LM of choice. Supports both API-based models and local models via vLLM. Additional arguments such as `temperature` and `samples_per_problem` can be found in the file itself.

**Outputs the original dataset with this additional column**:
- `samples`: List of model-generated responses for each problem

**Example Usage**:
```bash
python generate_reasoning_samples.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --dataset math \
    --output_path ../datasets/MATH500_with_Llama_3.1_8B_Instruct_samples.hf \
    --max_rows 10 \
    --samples_per_instruction 2 \
    --temperature 0.7
```

If nothing is passed for `--max_rows`, all problems in the dataset will be processed. If nothing is passed for `--samples_per_instruction`, 1 sample will be generated per problem.

**Currently Supported Datasets**:
- `mmlu`, `mmlu_pro`, `math`, `gpqa`, `gpqa_diamond`

**Currently Supported Models**:

API-Provider Models
- `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, `o3-mini-2025-01-31`, `o1-2024-12-17`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`, `deepseek-ai/DeepSeek-R1`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`

vLLM Models
- Any model supported by vLLM e.g `meta-llama/Llama-3.1-8B-Instruct`

### Step 2: Get Ground Truth Correctness of Samples
**Script**: [`unified_evaluation.py`](unified_evaluation.py)

Extracts final answers from the raw reasoning samples and checks them against ground truth labels. Uses a mixture of GPT-4o-mini and procedural Python code to robustly parse different answer formats (multiple choice letters, numerical values, mathematical expressions) and verify correctness.

**Outputs the existing dataset with these additional columns**:
- `extracted_answers`: List of final answers extracted from each sample
- `answer_correct`: List of boolean values indicating whether each extracted answer is correct

**Example Usage**:
```bash
python unified_evaluation.py \
    --input_path ../datasets/MATH500_with_Llama_3.1_8B_Instruct_samples.hf \
    --output_path ../datasets/MATH500_with_Llama_3.1_8B_Instruct_evaluated.hf \
    --dataset_type math
```

### Step 3: Score with Verifiers
**Script**: [`unified_RMs_and_LM_Judges.py`](unified_RMs_and_LM_Judges.py)

Collects scores from multiple reward models and LM judges for each sample. Includes 15+ reward models (like GRM, ArmorRM, URM) and supports various LM judges (GPT-4o, Claude, etc.). Manages GPU allocation automatically and can process verifiers either in parallel or sequentially depending on memory constraints.

**Outputs the existing dataset with these additional columns:**
- `{RewardModelName}_scores`: List of scores from each reward model (e.g., `GRM_scores`, `ArmorRM_scores`)
- `{LMJudgeName}_verdicts`: List of verdict scores from each LM judge (e.g., `GPT-4o-mini_verdicts`)
- For Process Reward Models (PRM):
  - `{PRMName}_min_scores`: Minimum scores across reasoning steps
  - `{PRMName}_max_scores`: Maximum scores across reasoning steps  
  - `{PRMName}_avg_scores`: Average scores across reasoning steps
  - `{PRMName}_step_scores`: Detailed step-by-step scores

**Example Usage**:
```bash
python unified_RMs_and_LM_Judges.py \
    --dataset_path ../datasets/MATH500_with_Llama_3.1_8B_Instruct_evaluated.hf \
    --output_path ../datasets/MATH500_with_Llama_3.1_8B_Instruct.hf \
    --reward_models GRM,ArmorRM,URM,QRM \
    --lm_judges Llama-3.1-8B-Instruct-Together,GPT-4o-mini \
    --batch_size 4
```

**For a complete list of available verifiers, see [Verifiers →](VERIFIERS.md)**

**Note**:
If you encounter a GLIBC version error (`GLIBC_2.32 not found`), this is due to the `flash-attn` package requiring a newer GLIBC than available on Ubuntu 20.04. Datasets can be generated without flash attention simply by running `pip uninstall flash-attn`, which will cause the script to use standard attention instead. We also recommend you uncomment `"attn_implementation": "eager"` in this case to use eager attention.

## Resource Requirements

- **Compute**: Small models (<=8B parameters) can be run on a single H100 while larger models like INFORM (70B) and Qwen72B require tensor parallelism across multiple H100s
- **Storage**: ~1-10GB per dataset depending on size and number of samples

## Inference Calls Formula

Here's how to estimate the total number of inference calls for each pipeline step:

### Step 1: Generate Samples
```
min(len(dataset), max_rows) × samples_per_instruction
```

### Step 2: Get Ground Truth Correctness
**Answer Extraction (all datasets):**
```
min(len(dataset), max_rows) × min(samples_per_instruction, max_samples)
```

**Answer Verification (dataset-dependent):**
- **MMLU, MMLU-Pro, GPQA:** `0` (string matching only)
- **MATH:** `min(len(dataset), max_rows) × min(samples_per_instruction, max_samples)`

### Step 3: Score with Verifiers
**Reward Models:**
```
min(len(dataset), max_rows) × min(samples_per_instruction, max_samples) × num_reward_models
```

**LM Judges:**
```
min(len(dataset), max_rows) × min(samples_per_instruction, max_samples) × num_lm_judges × verdicts_per_sample
```

### Parameters
- `len(dataset)`: Number of problems (rows) in the dataset
- `max_rows`: Limit on dataset rows to process
- `max_samples`: Limit on samples per instruction  
- `samples_per_instruction`: Number of model responses per problem
- `num_reward_models`: Count of reward models used
- `num_lm_judges`: Count of LM judges used
- `verdicts_per_sample`: Verdicts per sample (default: 1)

## Final Dataset Format

All datasets share the following **core structure**:

```javascript
{
  'instruction': 'Question: What is the capital of France?...',  // full prompt used for inference
  'samples': [
    'Looking at this question, I need to...',  // model response 1
    'To solve this, I first consider...',      // model response 2
    // ... more responses (typically 32 per problem)
  ],
  'extracted_answers': ['B', 'A', 'B', ...],   // extracted from each sample
  'answer_correct': [True, False, True, ...],  // ground truth correctness
  'answer': 'B',                               // ground truth answer
  
  // Reward model scores (one score per sample)
  'GRM_scores': [0.85, 0.23, 0.91, ...],
  'ArmorRM_scores': [0.78, 0.15, 0.88, ...],
  // ... additional reward models vary by dataset
  
  // LM judge verdicts (nested lists)
  'DeepSeekLlama70B_verdicts': [[1.0], [0.0], [1.0], ...],
  'Llama-3.3-70B-Instruct_verdicts': [[1.0], [0.0], [1.0], ...],
  // ... additional LM judges vary by dataset
}
```

In addition to the core structure, our datasets have several additional columns related to the various types of benchmark problems. Below, we explain each of the additional columns that you may find in our released datasets.

**MATH500:**
- `problem`: Raw math problem (same as `instruction`)
- `solution`: Ground truth solution steps  
- `subject`: Math subject area (e.g., "Algebra", "Geometry")
- `level`: Difficulty level (1-5)
- `unique_id`: Problem identifier

**GPQA:**
- `original_instruction`: Original problem statement
- `type`: "diamond" or "main" 
- `options`: Multiple choice options array
- `correct_answer`: Same as `answer`
- `subject`: Always "science"

**MMLU:**
- `question`: Core question without choices or instructions
- `problem`: Same as `instruction` (question + choices + solve instructions)
- `choices`: Multiple choice options array
- `subject`: Academic subject (e.g., "college_medicine", "college_mathematics")

**MMLU-Pro:**
- `question`: Core question without choices or instructions  
- `problem`: Same as `instruction`
- `options`: Multiple choice options array (instead of `choices`)
- `answer_index`: Index of correct answer in options array
- `subject`: Broad subject area (e.g., "history", "biology", "math", "physics")

**Note:**
Our released datasets also have a `weaver_scores` column which contain the scores produced by running the best configuration of Weaver. For more information check out [selection/README.md](../selection/README.md).
