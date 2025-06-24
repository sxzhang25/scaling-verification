from collections import defaultdict


DATASET_TO_HF = {
    'MATH-500': {
        '8B': 'hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1',
        '70B': 'hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1'
    },
    'GPQA': {
        '8B': 'hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1',
        '70B': 'hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1'
    },
    # NOTE: using GPQA-Diamond as dataset enables downstream filtering of dataset to diamond problems only
    'GPQA-Diamond': {
        '8B': 'hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1',
        '70B': 'hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1'
    },
    'MMLU': {
        '8B': 'hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1',
        '70B': 'hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1'
    },
    'MMLU-Pro': {
        '8B': 'hazyresearch/MMLU-Pro_with_Llama_3.1_8B_Instruct_v1',
        '70B': 'hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1'
    },
    'GPQA-1K': {
        '70B': 'hazyresearch/GPQA_Diamond_with_Llama_3.1_70B_Instruct_up_to_1K_Samples_v1'
    },
}


REWARD_MODELS_NAME_MAP = {
    # ArmorRM Variations
    'armor_rm_score': 'ArmorRM (8B)',
    'ArmorRM_scores': 'ArmorRM (8B)',
    'armor_rm_correctness': 'ArmorRM_correctness (8B)',
    'ArmorRM_correctness': 'ArmorRM_correctness (8B)',

    # Decision Tree Reward Variations
    'DecisionTreeReward8B_scores': 'DecisionTreeReward (8B)',
    'DecisionTreeReward27B_scores': 'DecisionTreeReward Gemma (27.2B)',

    # Eurus Variations
    'eurus_prm_scores': 'Eurus_PRM (7.62B)',
    'eurus_prm2_scores': 'Eurus_PRM2 (7.62B)',
    'EurusPRMStage1_avg_scores': 'EurusPRMStage1_avg_scores (7.62B)',
    'EurusPRMStage1_max_scores': 'EurusPRMStage1_max_scores (7.62B)',
    'EurusPRMStage1_min_scores': 'EurusPRMStage1_min_scores (7.62B)',
    'EurusPRMStage2_avg_scores': 'EurusPRMStage2_avg_scores (7.62B)',
    'EurusPRMStage2_max_scores': 'EurusPRMStage2_max_scores (7.62B)',
    'EurusPRMStage2_min_scores': 'EurusPRMStage2_min_scores (7.62B)',
    "EurusPRMStage1_step_scores": "EurusPRMStage1_step_scores (7.62B)",
    "EurusPRMStage2_step_scores": "EurusPRMStage2_step_scores (7.62B)",

    # GPM Variations
    'gpm_scores': 'GPM (8B)',
    'GPM_scores': 'GPM (8B)',

    # GRM Variations
    'grm_scores': 'GRM (8B)',
    'GRM_scores': 'GRM (8B)',
    'GRMGemma_scores': 'GRM_GEMMA (2B)',
    'grm_gemma_scores': 'GRM_GEMMA (2B)',

    'GRMLlama32_scores': 'GRM_LLama32 (3B)',
    'grm_llama32_scores': 'GRM_LLama32 (3B)',

    # Inform Variations
    'inform_scores': 'Inform (69.6B)',
    'INFORM_scores': 'Inform (69.6B)',

    # InternLM Variations
    'internlm_scores': 'InternLM2 (7B)',
    'InternLM_scores': 'InternLM2 (7B)',
    'internlm2_scores': 'InternLM2 (7B)',
    'InternLM2Reward7B_scores': 'InternLM2 (7B)',
    'InternLM2RewardModel_scores': 'InternLM2 (20B)',

    # LDL Reward Variations
    'LDLRewardGemma_scores': 'LDLReward (27.2B)',

    # OffsetBias Variations
    'offset_bias_scores': 'OffsetBias (8B)',
    'OffsetBias_scores': 'OffsetBias (8B)',

    # QRM Variations
    'qrm_scores': 'QRM (8B)',
    'QRM_scores': 'QRM (8B)',
    'qrm_gemma_scores': 'QRM_Gemma (27.2B)',
    'QRMGemma_scores': 'QRM_Gemma (27.2B)',
    'qrm_gemma_correctness': 'QRM_Gemma_correctness (27.2B)',

    # Qwen Variations
    'qwen25_math_scores': 'QwenPRM_avg_scores (7.63B)',
    'QwenPRM_avg_scores': 'QwenPRM_avg_scores (7.63B)',
    'QwenPRM_max_scores': 'QwenPRM_max_scores (7.63B)',
    'QwenPRM_min_scores': 'QwenPRM_min_scores (7.63B)',
    "QwenPRM_step_scores": "QwenPRM_step_scores (7.63B)",
    'Qwen72B_scores': 'Qwen72B_scores (72B)',

    # Skyworks Variations
    'skyworks_scores': 'Skyworks (8B)',
    'Skyworks_scores': 'Skyworks (8B)',
    'skywork_gemma_scores': 'Skyworks_Gemma (27.2B)',
    'SkyworksGemma_scores': 'Skyworks_Gemma (27.2B)',

    # SkyT1 Variations
    'SkyT1_verdicts': 'SkyT1 (32B)',

    # URM Variations
    'urm_scores': 'URM (8B)',
    'URM_scores': 'URM (8B)',

}


JUDGE_NAME_MAP = {
    # Claude 3.5 Haiku
    'judge_claude-3-5-haiku-latest_verdicts_v1': 'Claude-3.5-Haiku_v1',
    'Claude-3-5-Haiku_verdicts': 'Claude-3.5-Haiku',

    # Claude 3.5 Sonnet
    'judge_claude-3-5-sonnet-latest_verdicts': 'Claude-3.5-Sonnet',
    'claude-3-5-sonnet-latest_verdicts_v1': 'Claude-3.5-Sonnet_v1',
    'Claude-3-7-Sonnet_verdicts': 'Claude-3.7-Sonnet',

    # DeepSeek Variations
    'DeepSeekLlama70B_verdicts': 'DeepSeekLlama (70B)',
    'DeepSeekQwen32B_verdicts': 'DeepSeekQwen (32B)',

    # Gemma Judges
    'judge_gemma-2-27b-it_verdicts': 'Gemma-2-27B',

    # GPT 4o
    'judge_gpt-4o_verdicts': 'GPT-4o',
    'gpt-4o_verdicts_v1': 'GPT-4o_v1',
    'GPT-4o_verdicts': 'GPT-4o',

    # GPT 4o Mini
    'GPT-4o-mini_verdicts_v1': 'GPT-4o-mini_v1',
    'GPT-4o-mini_verdicts': 'GPT-4o-mini',
 
     # Llama Judges
    'Llama-3.3-70B-Instruct_verdicts': 'Llama-3.3-70B Instruct',
    'Llama-3.1-8B-Instruct-Together_verdicts': 'Llama-3.1-8B Instruct (Together)',
    'Llama-3.1-70B-Instruct-Together_verdicts': 'Llama-3.1-70B Instruct (Together)',

    'judge_llama-3.1-nemotron-70b-instruct-hf_verdicts': 'Llama-3.1 Nemotron 70B Instruct',
    'judge_meta-llama-3.1-405b-instruct-turbo_verdicts': 'Llama-3.1-405B Instruct Turbo',
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo_verdicts_v1': 'Llama-3.1-405B Instruct Turbo',

    'judge_llama-3.3-70b-instruct-turbo_verdicts': 'Llama-3.3-70B Instruct Turbo',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo_verdicts_v1': 'Llama-3.3-70B Instruct Turbo v1',

    'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts': 'Llama-3.1-405B Instruct Quantized',

    # Mixtral Judges
    'judge_nous-hermes-2-mixtral-8x7b-dpo_verdicts': 'Nous Hermes2 Mixtral-8x7B DPO',
    'judge_mixtral-8x22b-instruct-v0.1_verdicts': 'Mixtral-8x22B Instruct v.01',
    'Mixtral-8x22B-Instruct-v0.1_verdicts': 'Mixtral-8x22B',

    # Qwen Judges
    'judge_qwen2-72b-instruct_verdicts': 'Qwen2-72B Instruct',
    'Qwen/Qwen2.5-72B-Instruct_verdicts': 'Qwen2.5-72B Instruct',
    'judge_qwen2.5-72b-instruct-turbo_verdicts': 'Qwen2.5-72B Instruct Turbo',
    'Qwen/Qwen2.5-72B-Instruct-Turbo_verdicts_v1': 'Qwen2.5-72B Instruct Turbo v1',
    'judge_qwq-32b-preview_verdicts': 'QwQ-32B',
    'QwQ-32B-Preview_verdicts': 'QwQ-32B',

    # Other Judges
    'judge_wizardlm-2-8x22b_verdicts': 'WizardLM-2-8x22B',
    'WizardLM-2-8x22B_verdicts': 'WizardLM-2-8x22B',
}


# Combine reward and judge name maps
VERIFIER_NAME_MAP = {**REWARD_MODELS_NAME_MAP, **JUDGE_NAME_MAP}


VERIFIER_DESCRIPTIONS = {
    # ArmorRM Variations
    "ArmorRM (8B)": {"num_parameters": 7.51, "type": "reward", "access_type": "open_source", "link":"https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1"},
    "ArmorRM_correctness (8B)": {"num_parameters": 7.51, "type": "reward", "access_type": "open_source", "link":"https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1"},

    # Decision Tree Reward
    "DecisionTreeReward (8B)": {"num_parameters": 7.5, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/RLHFlow/Decision-Tree-Reward-Llama-3.1-8B"},
    "DecisionTreeReward Gemma (27.2B)": {"num_parameters": 27.2, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/RLHFlow/Decision-Tree-Reward-Gemma-2-27B"},

    # Eurus Variations
    "Eurus_PRM (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source"},
    "Eurus_PRM2 (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source"},
    "EurusPRMStage1_avg_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage1"},
    "EurusPRMStage1_max_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage1"},
    "EurusPRMStage1_min_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage1"},
    "EurusPRMStage1_step_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage1"},
    "EurusPRMStage2_avg_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage2"},
    "EurusPRMStage2_max_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage2"},
    "EurusPRMStage2_min_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage2"},
    "EurusPRMStage2_step_scores (7.62B)": {"num_parameters": 7.62, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/PRIME-RL/EurusPRM-Stage2"},

    # GPM Variations
    "GPM (8B)": {"num_parameters": 8.03, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/general-preference/GPM-Llama-3.1-8B"},

    # GRM Variations
    "GRM (8B)": {"num_parameters": 7.5, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Ray2333/GRM-Llama3-8B-rewardmodel-ft"},
    "GRM_GEMMA (2B)": {"num_parameters": 2.61, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Ray2333/GRM-Gemma2-2B-rewardmodel-ft"},
    "GRM_LLama32 (3B)": {"num_parameters": 3.21, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"},

    # Inform Variations
    "Inform (69.6B)": {"num_parameters": 69.6, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/infly/INF-ORM-Llama3.1-70B"},

    # InternLM Variations
    "InternLM (7B)": {"num_parameters": 7.36, "type": "reward", "access_type": "open_source", "link": ""},
    "InternLM2 (7B)": {"num_parameters": 7.36, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/internlm/internlm2-7b-reward"},
    "InternLM2 (20B)": {"num_parameters": 19.3, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/internlm/internlm2-20b-reward"},

    # LDL Reward Variations
    "LDLReward (27.2B)": {"num_parameters": 27.2, "type": "reward", "access_type": "open_source"},

    # OffsetBias Variations
    "OffsetBias (8B)": {"num_parameters": 7.5, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/NCSOFT/Llama-3-OffsetBias-RM-8B"},

    # QRM Variations
    "QRM (8B)": {"num_parameters": 7.51, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/nicolinho/QRM-Llama3.1-8B-v2"},

    "QRM_Gemma (27.2B)": {"num_parameters": 27.2, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/nicolinho/QRM-Gemma-2-27B"},
    "QRM_Gemma_correctness (27.2B)": {"num_parameters": 27.2, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/nicolinho/QRM-Gemma-2-27B"},


    # Qwen Variations
    "Qwen25_math_scores (7.63B)": {"num_parameters": 7.63, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B"},
    "QwenPRM_avg_scores (7.63B)": {"num_parameters": 7.63, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B"},
    "QwenPRM_max_scores (7.63B)": {"num_parameters": 7.63, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B"},
    "QwenPRM_min_scores (7.63B)": {"num_parameters": 7.63, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B"},
    "QwenPRM_step_scores (7.63B)": {"num_parameters": 7.63, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B"},

    "Qwen72B_scores (72B)": {"num_parameters": 72.8, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B"},

    # Skyworks Variations: which one is it 
    "Skyworks (8B)": {"num_parameters": 7.5, "type": "reward", "access_type": "open_source", "link":"https://huggingface.co/skywork/Skywork-Reward-Llama-3.1-8B-v0.2"},
    "Skyworks_Gemma (27.2B)": {"num_parameters": 27.2, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2"},

    # SkyT1 Variations
    "SkyT1 (32B)": {"num_parameters": 32.8, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview"},

    # URM Variations
    "URM (8B)": {"num_parameters": 7.54, "type": "reward", "access_type": "open_source", "link": "https://huggingface.co/LxzGordon/URM-LLaMa-3.1-8B"},

    # ------------------------------------------------------------------
    # Judge Variations
    # ------------------------------------------------------------------
    
    # Claude 3.5 Haiku
    "Claude-3.5-Haiku": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},
    'Claude-3.5-Haiku_v1': {"num_parameters": None, "type": "judge", "access_type": "closed_source"},

    # Claude 3.5 Sonnet
    "Claude-3.5-Sonnet": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},
    "Claude-3.5-Sonnet_v1": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},
    "Claude-3.7-Sonnet": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},

    # DeepSeek Variations
    "DeepSeekLlama (70B)": {"num_parameters": 70.0, "type": "judge", "access_type": "open_source"},
    "DeepSeekQwen (32B)": {"num_parameters": 32.0, "type": "judge", "access_type": "open_source"},

    # GPT 4o
    "GPT-4o": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},
    "GPT-4o_v1": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},

    # GPT 4o Mini
    "GPT-4o-mini": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},
    "GPT-4o-mini_v1": {"num_parameters": None, "type": "judge", "access_type": "closed_source"},

    # Llama variations:
    "Llama-3.3-70B Instruct": {"num_parameters": 70.0, "type": "judge", "access_type": "open_source"},
    "Llama-3.1-70B Instruct (Together)": {"num_parameters": 70.0, "type": "judge", "access_type": "open_source"},
    "Llama-3.1-8B Instruct (Together)": {"num_parameters": 8.0, "type": "judge", "access_type": "open_source"},

    "Llama-3.1 Nemotron 70B Instruct": {"num_parameters": 70.0, "type": "judge", "access_type": "open_source"},
    "Llama-3.1-405B Instruct Turbo": {"num_parameters": 405, "type": "judge", "access_type": "closed_source"},

    "Llama-3.3-70B Instruct Turbo": {"num_parameters": 70.0, "type": "judge", "access_type": "closed_source"},
    "Llama-3.1-405B-Instruct-Turbo_verdicts_v1": {"num_parameters": 405.0, "type": "judge", "access_type": "open_source"},
    "Llama-3.1-405B Instruct Quantized": {"num_parameters": 405.0, "type": "judge", "access_type": "open_source"},

    # Mistral Variations
    "Nous Hermes2 Mixtral-8x7B DPO": {"num_parameters": 46.7, "type": "judge", "access_type": "open_source", "link":"https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"},
    "Mixtral-8x22B": {"num_parameters": 141.0, "type": "judge", "access_type": "open_source", "link": "https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1"},
    "Mixtral-8x22B Instruct v.01": {"num_parameters": 141.0, "type": "judge", "access_type": "open_source"},

    # Qwen 2.5 72B Instruct
    "Qwen2-72B Instruct": {"num_parameters": 72.7, "type": "judge", "access_type": "open_source", "link":"https://huggingface.co/Qwen/Qwen2-72B-Instruct"},
    "Qwen2.5-72B Instruct": {"num_parameters": 72.7, "type": "judge", "access_type": "open_source", "link":"https://huggingface.co/Qwen/Qwen2.5-72B-Instruct"},
    "Qwen2.5-72B Instruct Turbo": {"num_parameters": 72.7, "type": "judge", "access_type": "open_source"},
    "Qwen2.5-72B Instruct Turbo v1": {"num_parameters": 72.7, "type": "judge", "access_type": "open_source"},
    "QwQ-32B" : {"num_parameters": 32.0, "type": "judge", "access_type": "open_source"},

    # Judge Models (Closed Source)
    "WizardLM-2-8x22B": {"num_parameters": 141.0, "type": "judge", "access_type": "open_source", "link": "https://huggingface.co/alpindale/WizardLM-2-8x22B"},
}


DATASET_TO_REWARD_MODELS = {
    'hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1': [
        'ArmorRM_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'InternLM2Reward7B_scores',
        'OffsetBias_scores',
        'QRM_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'Skyworks_scores',
        'URM_scores'
    ],
    'hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1': [
        'ArmorRM_scores',
        'DecisionTreeReward27B_scores',
        'DecisionTreeReward8B_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'INFORM_scores',
        'InternLM2Reward7B_scores',
        'InternLM2RewardModel_scores',
        'LDLRewardGemma_scores',
        'OffsetBias_scores',
        'QRMGemma_scores',
        'QRM_scores',
        'Qwen72B_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'SkyworksGemma_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1': [
        'ArmorRM_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'InternLM2Reward7B_scores',
        'OffsetBias_scores',
        'QRM_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1': [
        'ArmorRM_scores',
        'DecisionTreeReward27B_scores',
        'DecisionTreeReward8B_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'INFORM_scores',
        'InternLM2Reward7B_scores',
        'InternLM2RewardModel_scores',
        'LDLRewardGemma_scores',
        'OffsetBias_scores',
        'QRMGemma_scores',
        'QRM_scores',
        'Qwen72B_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'SkyworksGemma_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1': [
        'ArmorRM_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'InternLM2Reward7B_scores',
        'OffsetBias_scores',
        'QRM_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1': [
        'ArmorRM_scores',
        'DecisionTreeReward27B_scores',
        'DecisionTreeReward8B_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'INFORM_scores',
        'InternLM2Reward7B_scores',
        'InternLM2RewardModel_scores',
        'LDLRewardGemma_scores',
        'OffsetBias_scores',
        'QRMGemma_scores',
        'QRM_scores',
        'Qwen72B_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'SkyworksGemma_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/MMLU-Pro_with_Llama_3.1_8B_Instruct_v1': [
        'ArmorRM_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'InternLM2Reward7B_scores',
        'OffsetBias_scores',
        'QRM_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1': [
        'ArmorRM_scores',
        'DecisionTreeReward27B_scores',
        'DecisionTreeReward8B_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_step_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_step_scores',
        'GPM_scores',
        'GRMGemma_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'INFORM_scores',
        'InternLM2Reward7B_scores',
        'InternLM2RewardModel_scores',
        'LDLRewardGemma_scores',
        'OffsetBias_scores',
        'QRMGemma_scores',
        'QRM_scores',
        'Qwen72B_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_max_scores',
        'QwenPRM_min_scores',
        'QwenPRM_step_scores',
        'SkyworksGemma_scores',
        'Skyworks_scores',
        'URM_scores',
    ],
    'hazyresearch/GPQA_Diamond_with_Llama_3.1_70B_Instruct_up_to_1K_Samples_v1': [
        'QRMGemma_scores',
        'INFORM_scores',
        'Qwen72B_scores',
        'LDLRewardGemma_scores',
        'InternLM2RewardModel_scores',
        'QRM_scores',
        'InternLM2Reward7B_scores',
        'GRMGemma_scores',
        'QwenPRM_min_scores',
        'QwenPRM_max_scores',
        'QwenPRM_avg_scores',
        'QwenPRM_step_scores',
        'EurusPRMStage2_min_scores',
        'EurusPRMStage2_max_scores',
        'EurusPRMStage2_avg_scores',
        'EurusPRMStage2_step_scores',
        'Skyworks_scores',
        'ArmorRM_scores',
        'OffsetBias_scores',
        'GRMLlama32_scores',
        'GRM_scores',
        'URM_scores',
        'GPM_scores',
        'EurusPRMStage1_min_scores',
        'EurusPRMStage1_max_scores',
        'EurusPRMStage1_avg_scores',
        'EurusPRMStage1_step_scores'
    ]
}


DATASET_TO_LM_JUDGES = {
    'hazyresearch/MATH-500_with_Llama_3.1_8B_Instruct_v1': [],
    'hazyresearch/MATH500_with_Llama_3.1_70B_Instruct_v1': [
        'DeepSeekLlama70B_verdicts',
        'DeepSeekQwen32B_verdicts',
        'Llama-3.3-70B-Instruct_verdicts',
        'Mixtral-8x22B-Instruct-v0.1_verdicts',
        'Qwen/Qwen2.5-72B-Instruct_verdicts',
        'SkyT1_verdicts',
        'WizardLM-2-8x22B_verdicts',
        'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts',
    ],
    'hazyresearch/GPQA_with_Llama_3.1_8B_Instruct_v1': [],
    'hazyresearch/GPQA_with_Llama_3.1_70B_Instruct_v1': [
        'WizardLM-2-8x22B_verdicts',
        'Mixtral-8x22B-Instruct-v0.1_verdicts',
        'Qwen/Qwen2.5-72B-Instruct_verdicts',
        'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts',
        'DeepSeekQwen32B_verdicts',
        'SkyT1_verdicts',
        'DeepSeekLlama70B_verdicts',
        'Llama-3.3-70B-Instruct_verdicts',
    ],
    'hazyresearch/MMLU_with_Llama_3.1_8B_Instruct_v1': [],
    'hazyresearch/MMLU_with_Llama_3.1_70B_Instruct_v1': [
        'DeepSeekLlama70B_verdicts',
        'DeepSeekQwen32B_verdicts',
        'Llama-3.3-70B-Instruct_verdicts',
        'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts',
        'Mixtral-8x22B-Instruct-v0.1_verdicts',
        'Qwen/Qwen2.5-72B-Instruct_verdicts',
        'SkyT1_verdicts',
        'WizardLM-2-8x22B_verdicts',
    ],
    'hazyresearch/MMLU-Pro_with_Llama_3.1_8B_Instruct_v1': [],
    'hazyresearch/MMLU-Pro_with_Llama_3.1_70B_Instruct_v1': [
        'DeepSeekLlama70B_verdicts',
        'DeepSeekQwen32B_verdicts',
        'Llama-3.3-70B-Instruct_verdicts',
        'Mixtral-8x22B-Instruct-v0.1_verdicts',
        'Qwen/Qwen2.5-72B-Instruct_verdicts',
        'SkyT1_verdicts',
        'WizardLM-2-8x22B_verdicts',
        'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts',
    ],
    "hazyresearch/GPQA_Diamond_with_Llama_3.1_70B_Instruct_up_to_1K_Samples_v1": [
        'DeepSeekLlama70B_verdicts',
        'DeepSeekQwen32B_verdicts',
        'Llama-3.3-70B-Instruct_verdicts',
        'Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts',
        'Mixtral-8x22B-Instruct-v0.1_verdicts',
        'Qwen/Qwen2.5-72B-Instruct_verdicts',
        'SkyT1_verdicts',
        'WizardLM-2-8x22B_verdicts',
    ]
}


def get_dataset_to_verifiers():
    """
    Get a dictionary mapping from dataset names to verifiers.
    """
    DATASET_TO_VERIFIERS = defaultdict(list)

    for key, value in DATASET_TO_REWARD_MODELS.items():
        DATASET_TO_VERIFIERS[key].extend(value)

    for key, value in DATASET_TO_LM_JUDGES.items():
        DATASET_TO_VERIFIERS[key].extend(value)

    # Convert back to a normal dictionary if needed:
    DATASET_TO_VERIFIERS = dict(DATASET_TO_VERIFIERS)
    return DATASET_TO_VERIFIERS

DATASET_TO_VERIFIERS = get_dataset_to_verifiers()
