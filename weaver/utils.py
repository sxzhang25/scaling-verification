import numpy as np
import sys
import os 
# Local imports: Update to your local environment if needed
sys.path.insert(0, "../")
from weaver.config_handler import VerifierConfig
from main import load_dataset_and_metrics, apply_verifiers
from weaver.constants import REWARD_MODELS_NAME_MAP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# cache dir is relative to the parent of the current dir
CACHE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..","cache/"))
METRICS_DIR = os.path.abspath(os.path.join(CURRENT_DIR,"metrics2/"))

###############################################################################
# Utility Functions
#

def preprocess_score(var_array, verifier_name, threshold=None):
    """
    Convert raw scores to [0..1] for RMs; judges assumed 0/1 already.
    Optionally binarize if threshold is specified.
    """
    arr = np.array(var_array, dtype=float)

    old_rm_names = list(REWARD_MODELS_NAME_MAP.keys())
    new_rm_names = list(REWARD_MODELS_NAME_MAP.values())

    rm_names = old_rm_names + new_rm_names

    is_rm = verifier_name in rm_names

    # Min-max normalization
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if (not is_rm) and (min_val >= 0.0) and (max_val <= 1.0):
        # Judge -> assume already 0 or 1
        return arr

    if min_val < max_val:
        arr = (arr - min_val) / (max_val - min_val)

    if threshold is not None:
        arr = (arr >= threshold).astype(float)

    assert np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0, f"verifier: {verifier_name}, min: {np.nanmin(arr)}, max: {np.nanmax(arr)}"
    return arr


def get_best_ensemble_answer_p_problem(dataset_tmp, reward_threshold=0.5, mv_as_voter=1):
    """
    Per-problem logistic regression ensemble (one LR per problem).
    """
    enabled_verifiers = [
        VerifierConfig(
            name="LogisticRegressionEnsemble",
            enabled=True,
            params={
                "filter_strategy": "top_k",
                "filter_strategy_param": 1,
                "per_problem": 1,
                "mv_as_voter": mv_as_voter,
            })
    ]
    global_params = {
        "max_rows": None,
        "verbose": False,
        "normalization": "global",
        "verifier_subset": None,  # Will use all available RMs for that dataset
    }
    base_config = {
        "global_params": global_params,
        "verifiers": []
    }
    dataset, verifier_metrics = load_dataset_and_metrics(dataset_tmp,
                                                         reward_threshold,
                                                         global_params,
                                                         mv_as_voter,
                                                         metrics_dir=METRICS_DIR)

    _, verification_mask = apply_verifiers(
        dataset=dataset,
        verifiers=enabled_verifiers,
        verifier_metrics=verifier_metrics,
        base_config=base_config,
        dataset_path=dataset_tmp,
        use_cached_stages=False,
        cache_dir=CACHE_DIR,
        verifier_subset=global_params["verifier_subset"],
        return_mask=True,
        mv_as_voter=mv_as_voter,
    )
    return verification_mask


#############################################################################

def compute_pass_at_k_gt(true_labels, k_values, return_mean=True):
    """
    Compute Pass@k for a matrix of shape (num_problems x num_answers),
    where each element is binary correctness (0 or 1).
    """
    num_problems, num_answers = true_labels.shape
    pass_at_k_results = {}
    for k in k_values:
        if k <= num_answers:
            pass_k = np.any(true_labels[:, :k] == 1, axis=1)
            if return_mean:
                pass_at_k_results[k] = np.nanmean(pass_k)
            else:
                pass_at_k_results[k] = pass_k.astype(int)
    return pass_at_k_results


def calculate_majority_at_k(dataset, k: int, return_mean=True):
    """
    Majority@k:
      - For each problem, collect all answers and correctness.
      - Group identical answers.
      - Take the top-k most frequent answers.
      - If any of those top-k is correct by majority (>50%), 
        the problem is counted correct.
    Returns either mean accuracy or per-problem array.
    """
    correct_problems = 0
    total_problems = 0
    all_problems = np.zeros(len(dataset))

    for i in range(len(dataset)):
        try:
            answers = dataset['extracted_answers'][i]
            answer_correct = dataset['answer_correct'][i]
        except KeyError:
            answers = dataset['samples'][i]
            answer_correct = dataset['is_corrects'][i]
    
        answer_counts, answer_correctness = {}, {}
        for ans, is_correct in zip(answers, answer_correct):
            if ans == 'NO_ANSWER':
                continue
            if ans not in answer_counts:
                answer_counts[ans] = 0
                answer_correctness[ans] = []
            answer_counts[ans] += 1
            answer_correctness[ans].append(is_correct)

        if answer_counts:
            total_problems += 1
            top_k_answers = sorted(
                answer_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            for ans, _ in top_k_answers:
                # If this answer is correct by majority
                if sum(answer_correctness[ans]) > len(answer_correctness[ans]) / 2:
                    correct_problems += 1
                    all_problems[i] = 1
                    break

    return correct_problems / total_problems if return_mean else all_problems


def classify_difficulty(df_row):
    """
    Classify a problem into {Hard, Medium, Easy} based on mean correctness.
    Thresholds: <=0.1 -> Hard; >0.1 <=0.5 -> Medium; >0.5 -> Easy.
    """
    levels = ["Hard", "Medium", "Easy"]
    thresholds = [0.1, 0.5]
    mean_correct = np.stack(df_row).mean()
    return levels[np.digitize(mean_correct, thresholds, right=True)]


def get_accuracy(df_row):
    """
    Given a list/array of correctness (0/1), return mean.
    """
    return np.stack(df_row).mean()


###############################################################################
# Ensemble Functions (wrapping calls to `weaver`)
###############################################################################

def get_best_ensemble_answer(dataset_tmp, reward_threshold=0.5, mv_as_voter=1):
    """
    Chooses the ensemble method based on dataset name + model size.
    Returns a mask of shape (num_problems, num_answers) with exactly one chosen answer (top-1).

    Here all values are from Spreadsheet in Scaling Verification:
    Datasets 
    - MATH-500: 8B: LogisticRegressionEnsemble, 70B: NaiveEnsemble
    - MMLU-College: 8B: NaiveEnsemble, 70B: NaiveBayesEnsemblePerProblem
    - MMLU-Pro: 8B: LogisticRegressionEnsemble, 70B: LogisticRegressionEnsemble
    - GPQA: 8B: LogisticRegressionEnsemble, 70B: Placehodler
    """
    if ("MATH-500" in dataset_tmp) and ("8B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="LogisticRegressionEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": ["gpm_scores", "grm_scores", "internlm_scores"]
        }
    elif ("MATH500" in dataset_tmp) and ("70B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="NaiveEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": [
                "GRMLlama32_scores", 
                "LDLRewardGemma_scores", 
                "Qwen72B_scores"
            ]
        }
    elif ("MMLU-College" in dataset_tmp) and ("8B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="NaiveEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": ["internlm_scores", "offset_bias_scores", "qrm_scores"],
        }
    elif ("MMLU-College" in dataset_tmp) and ("70B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="NaiveBayesEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": ["ArmorRM_scores", "Qwen72B_scores", "Skyworks_scores"],
        }
    elif ("MMLU-Pro" in dataset_tmp) and ("8B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="LogisticRegressionEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": ["GPM_scores", "GRM_scores", "QwenPRM_avg_scores"]
        }
    elif ("MMLU-Pro" in dataset_tmp) and ("70B" in dataset_tmp):
        # Placeholder for real subsets
        enabled_verifiers = [
            VerifierConfig(
                name="LogisticRegressionEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": ["GPM_scores", "GRM_scores", "QwenPRM_avg_scores"]
        }
    elif ("GPQA" in dataset_tmp) and ("8B" in dataset_tmp):
        enabled_verifiers = [
            VerifierConfig(
                name="LogisticRegressionEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": [
                "ArmorRM_scores", 
                "InternLM2Reward7B_scores", 
                "URM_scores"
            ]
        }
    elif ("GPQA" in dataset_tmp) and ("70B" in dataset_tmp):
        # Placeholder for real subsets
        enabled_verifiers = [
            VerifierConfig(
                name="LogisticRegressionEnsemble",
                enabled=True,
                params={
                    "filter_strategy": "top_k",
                    "filter_strategy_param": 1,
                    "mv_as_voter": mv_as_voter,
                })
        ]
        global_params = {
            "max_rows": None,
            "verbose": False,
            "normalization": "global",
            "verifier_subset": [
                "ArmorRM_scores", 
                "InternLM2Reward7B_scores", 
                "URM_scores"
            ]
        }
    else:
        raise ValueError(f"Dataset not recognized for ensemble: {dataset_tmp}")

    base_config = {
        "global_params": global_params,
        "verifiers": []
    }
    # Load and apply verifiers
    dataset, verifier_metrics = load_dataset_and_metrics(dataset_tmp,
                                                         reward_threshold,
                                                         global_params,
                                                         mv_as_voter,
                                                         metrics_dir=METRICS_DIR)
    
    _, verification_mask = apply_verifiers(
        dataset=dataset,
        verifiers=enabled_verifiers,
        verifier_metrics=verifier_metrics,
        base_config=base_config,
        dataset_path=dataset_tmp,
        use_cached_stages=False,
        cache_dir=CACHE_DIR,
        verifier_subset=global_params["verifier_subset"],
        return_mask=True,
        mv_as_voter=mv_as_voter,
    )
    return verification_mask


