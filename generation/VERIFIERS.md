# Available Verifiers

This document lists all the reward models and LM judges available in the Weaver framework for scoring reasoning samples.

## Reward Models

| Model Name | Parameters | Type | Description |
|------------|------------|------|-------------|
| `GRM` | 8B | Standard RM | General Reward Model (Llama3-8B) |
| `ArmorRM` | 8B | Standard RM | ArmoRM-Llama3-8B |
| `URM` | 8B | Standard RM | Universal Reward Model (Llama3.1-8B) |
| `QRM` | 8B | Multi-aspect RM | Quality Reward Model (Llama3.1-8B) |
| `GPM` | 8B | General Preference | General Preference Model (Llama3.1-8B) |
| `GRMLlama32` | 3B | Standard RM | GRM-Llama3.2-3B |
| `OffsetBias` | 8B | Standard RM | Llama3-OffsetBias-RM-8B |
| `GRMGemma` | 2B | Standard RM | GRM-Gemma2-2B |
| `Skyworks` | 8B | Standard RM | Skywork-Reward-Llama-3.1-8B |
| `SkyworksGemma` | 27B | Standard RM | Skywork-Reward-Gemma-2-27B |
| `QRMGemma` | 27B | Multi-aspect RM | QRM-Gemma-2-27B |
| `LDLRewardGemma` | 27B | Standard RM | LDL-Reward-Gemma-2-27B |
| `QwenPRM` | 7B | Process RM | Qwen2.5-Math-PRM-7B (step-by-step scoring) |
| `Qwen72B` | 72B | Standard RM | Qwen2.5-Math-RM-72B |
| `Qwen72BPRM` | 72B | Process RM | Qwen2.5-Math-PRM-72B (step-by-step scoring) |
| `EurusPRMStage1` | 7B | Process RM | Eurus PRM Stage 1 (step-by-step scoring) |
| `EurusPRMStage2` | 7B | Process RM | Eurus PRM Stage 2 (step-by-step scoring) |
| `InternLM2RewardModel` | 20B | Standard RM | InternLM2-20B-Reward |
| `InternLM2Reward7B` | 7B | Standard RM | InternLM2-7B-Reward |
| `DecisionTreeReward8B` | 8B | Multi-aspect RM | Decision-Tree-Reward-Llama-3.1-8B |
| `DecisionTreeReward27B` | 27B | Multi-aspect RM | Decision-Tree-Reward-Gemma-2-27B |
| `INFORM` | 70B | Standard RM | INF-ORM-Llama3.1-70B |

### Reward Model Types

- **Standard RM**: Provides a single score per response
- **Multi-aspect RM**: Provides scores across multiple dimensions (helpfulness, correctness, etc.)
- **Process RM**: Provides step-by-step scores for reasoning chains, includes min/max/avg aggregations

## LM Judges

### Local Models (vLLM)

| Model Name | Parameters | Model Path |
|------------|------------|------------|
| `DeepSeekLlama70B` | 70B | deepseek-ai/DeepSeek-R1-Distill-Llama-70B |
| `DeepSeekQwen32B` | 32B | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| `SkyT1` | 32B | NovaSky-AI/Sky-T1-32B-Preview |
| `Llama-3.3-70B-Instruct` | 70B | meta-llama/Llama-3.3-70B-Instruct |
| `Meta-Llama-3.1-405B-Instruct-quantized.w8a16` | 405B | neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a16 |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | Qwen/Qwen2.5-72B-Instruct |
| `QwQ-32B` | 32B | Qwen/QwQ-32B |
| `WizardLM-2-8x22B` | 176B | alpindale/WizardLM-2-8x22B |
| `Mixtral-8x22B-Instruct-v0.1` | 176B | mistralai/Mixtral-8x22B-Instruct-v0.1 |
| `DeepSeekLlama8B` | 8B | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| `DeepSeekQwen7B` | 7B | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| `Llama-3.1-8B-Instruct` | 8B | meta-llama/Llama-3.1-8B-Instruct |
| `Gemma-3-12B-Instruct` | 12B | google/gemma-3-12b-it |
| `Gemma-3-4B-Instruct` | 4B | google/gemma-3-4b-it |
| `Phi-4-4B-Instruct` | 4B | microsoft/Phi-4-mini-instruct |
| `Qwen-2.5-7B-Instruct` | 7B | Qwen/Qwen2.5-7B-Instruct |
| `Qwen-2.5-Math-7B-Instruct` | 7B | Qwen/Qwen2.5-Math-7B-Instruct |
| `Mistral-7B-Instruct-v0.2` | 7B | mistralai/Mistral-7B-Instruct-v0.2 |

### API Models

| Model Name | Provider | Model Path |
|------------|----------|------------|
| `GPT-4o` | OpenAI | gpt-4o |
| `GPT-4o-mini` | OpenAI | gpt-4o-mini |
| `Claude-3-7-Sonnet` | Anthropic | claude-3-7-sonnet-latest |
| `Claude-3-5-Sonnet` | Anthropic | claude-3-5-sonnet-latest |
| `Claude-3-5-Haiku` | Anthropic | claude-3-5-haiku-latest |
| `Llama-3.1-8B-Instruct-Together` | Together AI | meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo |
| `Llama-3.1-70B-Instruct-Together` | Together AI | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |

## Usage Examples

### Using Reward Models Only
```bash
python unified_RMs_and_LM_Judges.py \
    --dataset_path ../datasets/MATH500_evaluated.hf \
    --output_path ../datasets/MATH500_with_scores.hf \
    --reward_models GRM,ArmorRM,URM,QwenPRM \
    --batch_size 4
```

### Using LM Judges Only
```bash
python unified_RMs_and_LM_Judges.py \
    --dataset_path ../datasets/MATH500_evaluated.hf \
    --output_path ../datasets/MATH500_with_scores.hf \
    --lm_judges GPT-4o-mini,Llama-3.1-8B-Instruct \
    --verdicts_per_sample 1
```

### Using Both Reward Models and LM Judges
```bash
python unified_RMs_and_LM_Judges.py \
    --dataset_path ../datasets/MATH500_evaluated.hf \
    --output_path ../datasets/MATH500_with_scores.hf \
    --reward_models GRM,QwenPRM,INFORM \
    --lm_judges GPT-4o-mini,Claude-3-5-Sonnet \
    --batch_size 2
```

## Notes

- **Process Reward Models** (PRM) like `QwenPRM`, `EurusPRMStage1`, `EurusPRMStage2`, and `Qwen72BPRM` provide additional step-by-step scoring columns
- **Multi-aspect Reward Models** like `QRM`, `QRMGemma`, and `DecisionTreeReward*` provide scores across multiple quality dimensions
- Large models (70B+) may require multiple GPUs or sequential processing using the `--sequential_rm_processing` flag
- API models require appropriate environment variables set (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`)