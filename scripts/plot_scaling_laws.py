"""
Plot coverage of a model in the oracle setting.
"""
from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(PROJECT_ROOT)
FIGURE_PATH = Path("figures")  # Changed to use local figures directory
os.makedirs(FIGURE_PATH, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
try:
    import weaver
except:
    sys.path.append(str(PROJECT_ROOT))
from weaver.dataset import VerificationDataset
from weaver.constants import DATASET_TO_HF
from weaver.utils_metrics import calculate_pass_k_unbiased, calculate_pass_k_gt, calculate_majority_M_at_k
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from weaver.models import MajorityVote, NaiveEnsemble, WeakSupervised
DATA_IGNORE_LIST = ["CodeContests_gonly", "GPQA-v2"]
ALL_DATASETS = list(set(DATASET_TO_HF.keys()) - set(DATA_IGNORE_LIST))
ALL_DATASETS = [d for d in ALL_DATASETS if "v2" in d]

ALL_MODEL_SIZES = list(set(DATASET_TO_HF[ALL_DATASETS[0]].keys()) - set(DATA_IGNORE_LIST))

# Add o3-mini baseline performances
O3_MINI_BASELINES = {
    "AIMO-v2": 83.33,
    "MATH-500-v2": 94.40,
    "GPQA-v2-Diamond": 73.99,
    "GPQA-1K-v2": 73.99,
    "MMLU-College-v2": 92.21,
    "MMLU-Pro-v2": 86.00,
    "BBH-v2": 94.10
}

# Add Llama70B MV baseline for 8B models
LLAMA70B_MV_BASELINES = {
    "AIMO-v2": 31.3,
    "MATH-500-v2": 78.0,
    "GPQA-v2-Diamond": 42.9,
    "GPQA-1K-v2": 42.9,
    "MMLU-College-v2": 82.6,
    "MMLU-Pro-v2": 69.9,
    "BBH-v2": 85.1
}

# Add cluster configurations for each dataset and model size
CLUSTER_CONFIGS = {
    "70B": {
        "AIMO-v2": {"n_clusters": 5},
        "MATH-500-v2": {"n_clusters": 5},
        "GPQA-v2-Diamond": {"n_clusters": 1},
        "GPQA-1K-v2": {"n_clusters": 1},
        "MMLU-College-v2": {"n_clusters": 1},
        "MMLU-Pro-v2": {"n_clusters": 1},
        "BBH-v2": {"n_clusters": 1}
    },
    "8B": {
        "AIMO-v2": {"n_clusters": 5},
        "MATH-500-v2": {"n_clusters": 5},
        "GPQA-v2-Diamond": {"n_clusters": 1},
        "MMLU-College-v2": {"n_clusters": 1},
        "MMLU-Pro-v2": {"n_clusters": 1},
        "BBH-v2": {"n_clusters": 1}
    }
}



# Dictionary of colors for each approach
# TODO: replace all places where we don't use the color config with the color config
COLOR_CONFIGS = {
    'Pass@k (Oracle Verification)': 'red',
    'Weak Supervision - Discrete': '#2ecc71',
    'Weak Supervision - Continuous': '#27ae60',
    'Naive Ensemble': '#9467bd',
    'Majority1@K': '#8c564b',
    'Weak Supervision - Discrete + Tensor Decomp': '#16a085',
    'Weak Supervision - Discrete - No Verifier Dropping': '#1f77b4',
    'Best Verifier': '#1f77b4',
    'Naive Ensemble': '#9467bd',    
    'First Sample': '#A0A0A0',
    'o3-mini': '#404040',
}

def get_verifier_cfg(verifier_type, verifier_size, verifier_subset):
    class VerifierConfig:
        def __init__(self):
            self.verifier_type: str = verifier_type
            self.verifier_size: str = verifier_size
            self.verifier_subset: str = verifier_subset
        
        def get(self, key, default=None):
            """ Mimic dictionary .get() behavior """
            return getattr(self, key, default)

    return VerifierConfig()


def get_data_cfg(verifier_size):
    # TODO: Add support: what is the tradeoff of 
    data_cfg = {
        "train_split": 1.0,
        "train_queries": 1,
        "train_samples": 1,
        "test_samples": 1, 
        "random_seed": 0,
        "nan_replacement": 0,
        "normalize_type": "all_problems",
        "normalize_method": "minmax",
        "closest_train_problem_method": "mean_verifier_distance",
        "closest_train_problem_metric_type": "euclidean",
        "verifier_cfg": get_verifier_cfg("all", verifier_size, []),
        "shuffle_train_full": True,
        "shuffle_samples": True,
        "mv_as_verifier": True,
        "normalize_params": {
            "tmp": None
        }
        
    }
    return data_cfg

class CBArgs:
    def __init__(self, class_balance):
        self.class_balance = class_balance

weak_supervision_cfg = {
    "use_continuous": False,
    "k":2,
    "seed": 0,
    "binarize_threshold": 0.5,
    "metric": "scores",
    "n_epochs": 1000,
    "mu_epochs": 1000,
    "log_train_every": 1000,
    "lr": 0.00001,
    "deps_density": 0.1,
    "use_deps": "none",
    "deps_data_fraction": 0.01,
    "use_label_on_test": True,
    "drop_imbalanced_verifiers": "all",
    "drop_k": 100,
    "cb_args": CBArgs(class_balance="labels"),
    "drop_imbalanced_fallback": False, # if we want to drop all, drop none
}

def get_weak_supervision_cfg(dataset_name, model_size):

    data_weak_supervision_cfg = {
        "8B": {
            "AIMO-v2": {
                "reward_threshold": 0.25,
                "drop_imbalanced_verifiers": "large",
            },
            "BBH-v2": {
                "reward_threshold": 0.05,
                "drop_imbalanced_verifiers": "all",
            },
            "GPQA-v2-Diamond": {
                "reward_threshold": 0.1,
                "drop_imbalanced_verifiers": "all",
            },
            "MMLU-Pro-v2": {
                "reward_threshold": 0.1,
                "drop_imbalanced_verifiers": "all",
            },
            "MMLU-College-v2": {
                "reward_threshold": 0.15,
                "drop_imbalanced_verifiers": "all",
            },
            "MATH-500-v2": {
                "reward_threshold": 0.25,
                "drop_imbalanced_verifiers": "all",
            },                     
        },
        "70B": {
            "AIMO-v2": {
                "reward_threshold": 0.05,
                "drop_imbalanced_verifiers": "large",
            },
            "BBH-v2": {
                "reward_threshold": 0.95,
                "drop_imbalanced_verifiers": "small",
            },
            "GPQA-v2-Diamond": {
                "reward_threshold": 0.05,
                "drop_imbalanced_verifiers": "all",
            },
            "GPQA-1K-v2": {
                "reward_threshold": 0.01,
                "drop_imbalanced_verifiers": "all",
            },
            "MMLU-Pro-v2": {
                "reward_threshold": 0.95,
                "drop_imbalanced_verifiers": "all",
            },
            "MMLU-College-v2": {
                "reward_threshold": 0.95,
                "drop_imbalanced_verifiers": "all",
            },
            "MATH-500-v2": {
                "reward_threshold": 0.95,
                "drop_imbalanced_verifiers": "small",
            },            
            }
    }
    updated_cfg = weak_supervision_cfg.copy()
    updated_cfg.update(data_weak_supervision_cfg[model_size][dataset_name])
    
    return updated_cfg

def get_weak_supervision_no_drop_cfg(dataset_name, model_size):
    """
    Get weak supervision config with no verifier dropping.
    Uses the same reward thresholds as the regular config but sets drop_imbalanced_verifiers to None.
    """
    cfg = get_weak_supervision_cfg(dataset_name, model_size)
    cfg["drop_imbalanced_verifiers"] = None
    return cfg

def get_weak_supervision_tensor_cfg(dataset_name, model_size):
    """
    Get weak supervision config with tensor decomposition.
    Uses the same drop_imbalanced_verifiers as the regular config but enables tensor decomposition.
    """
    cfg = get_weak_supervision_cfg(dataset_name, model_size)
    cfg["use_tensor_decomp"] = True
    cfg["drop_imbalanced_verifiers"] = None  # Don't drop any verifiers to ensure we have enough for tensor decomposition
    return cfg

base_model_cfg = {
    "model_type": "weak_supervision",
    "model_class": "cluster",
    "cluster_cfg": {
        "cluster_type": "unique_extracted_answers",
        "n_clusters": 1,
        "preserve_ties": True,
        "uniform_with_ties": False,
        "cluster_on_all": True,
    }
}

def get_clean_dataset_name(dataset_name):
    """Remove -v2 suffix from dataset names for display purposes."""
    return dataset_name.replace("-v2", "")


def plot_verifier_scaling(dataset_name, model_size, verifier_size, num_bootstrap_samples=1, seed=0, order="from_worst"):
    """
    Plot accuracy vs number of verifiers in the ensemble.
    Shows how accuracy changes as we add more verifiers to the ensemble.
    Do this for each method 
    """
    rng = np.random.default_rng(seed)
    data_cfg = get_data_cfg(verifier_size)
    dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
    
    # Get configs for both discrete and continuous WS
    model_cfg_discrete = get_weak_supervision_cfg(dataset_name, model_size)
    model_cfg_continuous = get_weak_supervision_cfg(dataset_name, model_size)
    model_cfg_continuous['use_continuous'] = True  # Enable continuous mode

    (x_train, y_train) = dataset.test_data
    train_answers = dataset.test_answers
    all_verifier_names = np.asarray(dataset.verifier_names)
    
    num_problems, num_responses, num_verifiers = x_train.shape
    
    # Get verifier selection@1 accuracy
    # top_index = np.argmax(x_train, 1) #num_problems, num_verifiers
    #selected_correctness = np.stack([y_train[i, idx] for i, idx in enumerate(top_index)]) #num_problems, num_verifier
    #selection_accuracy = np.mean(selected_correctness, 0) # (num_verifiers, )

    # Sort verifiers from best to worst
    verifier_scores = get_verifier_scores(x_train, y_train)

    if order == "from_best":
        verifier_indices = np.argsort(verifier_scores)
    elif order == "from_worst":
        verifier_indices = np.argsort(verifier_scores)[::-1]
    elif order == "random":
        verifier_indices = np.random.permutation(num_verifiers)
    else:
        raise ValueError(f"Invalid order: {order}")
    
    # number of verifiers 
    num_verifiers = len(verifier_indices)


    # Number of verifiers along which to scale
    k_values = np.linspace(1, num_verifiers, 5, dtype=int)


    # Bootstrap
    B = num_bootstrap_samples
    results_weak_supervision = np.zeros((B, len(k_values)))
    results_weak_supervision_continuous = np.zeros((B, len(k_values)))
    results_naive_ensemble = np.zeros((B, len(k_values)))

    for b in range(B):
        for v_idx, v in enumerate(k_values):
            # Get scores for selected verifiers
            selected_verifiers = verifier_indices[:v]
            scores = x_train[..., selected_verifiers]


            # Get naive Ensemble accuracy
            results_naive_ensemble[b, v_idx] = get_naive_ensemble_accuracy([num_responses], scores, y_train)[0]

            # Get weak supervision accuracy
            model_cfg_discrete_local = model_cfg_discrete.copy()
            model_cfg_continuous_local = model_cfg_continuous.copy()
            model_cfg_discrete_local['verifier_names'] = all_verifier_names[selected_verifiers]
            model_cfg_continuous_local['verifier_names'] = all_verifier_names[selected_verifiers]
            if v == 1:
                # cannot drop verifier 
                model_cfg_discrete_local['drop_imbalanced_verifiers'] = None
                model_cfg_continuous_local['drop_imbalanced_verifiers'] = None
            results_weak_supervision[b, v_idx] = vanilla_weak_supervision([num_responses], scores, y_train, model_cfg_discrete_local)[0]
            results_weak_supervision_continuous[b, v_idx] = vanilla_weak_supervision([num_responses], scores, y_train, model_cfg_continuous_local)[0]
    
    
    # Plot results:
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    
    
    # Add baseline for naive ensemble (all verifiers)
    mean = np.mean(results_naive_ensemble, axis=0)
    err = np.std(results_naive_ensemble, axis=0)
    plt.plot(k_values, mean, 'o--', label="Naive Ensemble", color='orange')
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)
    
    # Add baseline for naive ensemble (all verifiers)
    mean = np.mean(results_weak_supervision, axis=0)
    err = np.std(results_weak_supervision, axis=0)
    plt.plot(k_values, mean, 'o--', label="Weak Supervision - Discrete", color='#2ecc71')  # Bright green
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)
    
    # Add baseline for weak supervision (continuous)
    mean = np.mean(results_weak_supervision_continuous, axis=0)
    err = np.std(results_weak_supervision_continuous, axis=0)
    plt.plot(k_values, mean, 'o--', label="Weak Supervision - Continuous", color='#27ae60')  # Darker green
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)
    
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=12)

    plt.xticks(k_values)
    plt.xlabel('Number of Verifiers')
    plt.ylabel('Selection Accuracy')
    plt.title(f'{get_clean_dataset_name(dataset_name)} - {model_size} Models')
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"verifier_scaling_{dataset_name}_{model_size}_verifier{verifier_size}_{order}.png", bbox_inches='tight')
    plt.close()
    
    # Print results table
    print(f"\nVerifier Scaling Results for {dataset_name} ({model_size}):")
    print("-" * 100)
    print("Number of Verifiers".ljust(20) + "Accuracy (%)".ljust(15) + "Std Dev (%)")
    print("-" * 100)
    for v, m, e in zip(k_values, mean, err):
        print(f"{v}".ljust(20) + f"{m*100:10.1f}".ljust(15) + f"{e*100:10.1f}")
    print("-" * 100)
    print()



def plot_accuracy_heatmap_verifiers_vs_samples(dataset_name, model_size, verifier_size='all', seed=0, num_bootstrap_samples=1, order='from_best'):
    """
    Generate a heatmap showing selection accuracy as a function of
    number of verifiers and number of repeated generations (samples).
    """
    rng = np.random.default_rng(seed)
    data_cfg = get_data_cfg(verifier_size)
    dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
    all_verifier_names = np.asarray(dataset.verifier_names)


    # Get configs for both discrete and continuous WS
    model_cfg_discrete = get_weak_supervision_cfg(dataset_name, model_size)
    model_cfg_continuous = get_weak_supervision_cfg(dataset_name, model_size)
    model_cfg_continuous['use_continuous'] = True  # Enable continuous mode

    (x_train, y_train) = dataset.test_data

    num_problems, num_responses, num_verifiers = x_train.shape

    top_index = np.argmax(x_train, 1) #num_problems, num_verifiers
    selected_correctness = np.stack([y_train[i, idx] for i, idx in enumerate(top_index)]) #num_problems, num_verifier
    selection_accuracy = np.mean(selected_correctness, 0) # (num_verifiers, )

    # random baseline
    random_baseline = y_train.mean(axis=1).mean()  # avg correctness per sampled group

    # Determine verifier order
    verifier_scores = get_verifier_scores(x_train, y_train)
    #above_random_baseline = (selection_accuracy >= random_baseline)
    #print(f"Number of verifiers above random baseline: {above_random_baseline}")
    #x_train = x_train[...,above_random_baseline]
    #num_verifiers = x_train.shape[2]
    #verifier_scores = verifier_scores[above_random_baseline]
    #all_verifier_names = all_verifier_names[above_random_baseline]
    if dataset_name == "AIMO-v2":
        model_cfg_discrete['drop_imbalanced_fallback'] = True
        model_cfg_continuous['drop_imbalanced_fallback'] = True
        model_cfg_discrete['deps_data_fraction'] = 0.05
        model_cfg_continuous['deps_data_fraction'] = 0.05

    if dataset_name == "MATH-500-v2":
        model_cfg_discrete['drop_imbalanced_fallback'] = True
        model_cfg_continuous['drop_imbalanced_fallback'] = True


    if order == "from_best":
        verifier_indices = np.argsort(verifier_scores)
    elif order == "from_worst":
        verifier_indices = np.argsort(verifier_scores)[::-1]
    elif order == "random":
        verifier_indices = np.random.permutation(num_verifiers)
    else:
        raise ValueError(f"Invalid order: {order}")

    # Define x and y axes for the heatmap
    k_values = [2**i for i in range(0, min(8, int(np.log2(num_responses)) + 1))]  # #samples (e.g. [1, 2, ..., 128])
    k_values.append(num_responses)
    k_values = np.unique(k_values)
    v_values = np.linspace(1, num_verifiers, 10, dtype=int)                      # #verifiers (e.g. 10 linearly spaced values)

    # Initialize results
    results_naive_ensemble = np.zeros((num_bootstrap_samples, len(v_values), len(k_values)))
    results_weak_supervision = np.zeros((num_bootstrap_samples, len(v_values), len(k_values)))
    results_weak_supervision_continuous = np.zeros((num_bootstrap_samples, len(v_values), len(k_values)))

    for b in range(num_bootstrap_samples):
        # Number of verifiers
        for vi, v in enumerate(v_values):
            selected_verifiers = verifier_indices[:v]
            scores_v = x_train[..., selected_verifiers]

            print(f"\nRunning bootstrap sample {b}/{num_bootstrap_samples} for {vi}/{len(v_values)} verifier subsets \n for {dataset_name} {model_size} {verifier_size}")

            # Number of samples
            for ki, k in enumerate(k_values):
                if k > num_responses:
                    continue
                x_sampled = np.zeros((num_problems, k, v))
                y_sampled = np.zeros((num_problems, k))

                for i in range(num_problems):
                    idx = rng.choice(num_responses, size=k, replace=False)
                    x_sampled[i] = scores_v[i, idx]
                    y_sampled[i] = y_train[i, idx]

                # Get naive Ensemble accuracy
                results_naive_ensemble[b, vi, ki] = get_naive_ensemble_accuracy([k], x_sampled, y_sampled)[0]

                # Get weak supervision accuracy
                model_cfg_discrete_local = model_cfg_discrete.copy()
                model_cfg_continuous_local = model_cfg_continuous.copy()
                model_cfg_discrete_local['verifier_names'] = all_verifier_names[selected_verifiers]
                model_cfg_continuous_local['verifier_names'] = all_verifier_names[selected_verifiers]

                if v == 1:
                    # cannot drop verifier 
                    model_cfg_discrete_local['drop_imbalanced_verifiers'] = None
                    model_cfg_continuous_local['drop_imbalanced_verifiers'] = None
                results_weak_supervision[b, vi, ki] = vanilla_weak_supervision([k], x_sampled, y_sampled, model_cfg_discrete_local)[0]
                results_weak_supervision_continuous[b, vi, ki] = vanilla_weak_supervision([k], x_sampled, y_sampled, model_cfg_continuous_local)[0]
        

    # Plot heatmap # number of verifiers vs number of samples
    fig = plt.figure(figsize=(10, 6))
    # add subplot for each model type 

    results_naive_ensemble = results_naive_ensemble.mean(0)
    results_weak_supervision = results_weak_supervision.mean(0)
    results_weak_supervision_continuous = results_weak_supervision_continuous.mean(0)

    vmin = min(results_naive_ensemble.min(), results_weak_supervision.min(), results_weak_supervision_continuous.min())
    vmax = max(results_naive_ensemble.max(), results_weak_supervision.max(), results_weak_supervision_continuous.max())

    color_map = plt.cm.plasma
    titles = ['Naive Ensemble', 'WS - Discrete', 'WS - Continuous']
    results_list = [results_naive_ensemble, results_weak_supervision, results_weak_supervision_continuous]

        
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for ax, result, title in zip(axes, results_list, titles):
        im = ax.imshow(result, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap=color_map)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Number of Repeated Generations')
        ax.set_xticks(np.arange(len(k_values)))
        ax.set_yticks(np.arange(len(v_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticklabels(v_values)
        
        # Annotate cells with values
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                val = result[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color='white' if val < (vmin + vmax)/2 else 'black')

    axes[0].set_ylabel('Number of Verifiers')

    # Shared colorbar
    # Attach colorbar to last axis without overlapping
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)  # increase pad to avoid overlap
    cbar = fig.colorbar(im, cax=cax)
    tick_values = np.linspace(vmin, vmax, num=6)  # 6 ticks evenly spaced
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
    cbar.set_label('Selection Accuracy', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add shared title
    fig.suptitle(f'Accuracy Heatmap: {get_clean_dataset_name(dataset_name)} ({model_size})',
                fontsize=15, fontweight='bold', y=1)

    # Save and close
    fig_name = f"heatmap_verifier_vs_samples_{dataset_name}_{model_size}_verifier{verifier_size}_{order}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURE_PATH / fig_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {FIGURE_PATH / fig_name}")



def main(dataset_name, model_size, verifier_size, num_bootstrap_samples=1, seed=0, include_tensor_decomp=False, include_no_drop=False, include_naive_ensemble=False):
    """
    Main function that can either plot a single dataset or a grid of datasets.
    Plots number of generations vs selection accuracy.
    """
    if isinstance(dataset_name, list):
        # If multiple datasets are provided, create a grid plot
        plot_grid(dataset_name, model_size, verifier_size, num_bootstrap_samples, seed, include_tensor_decomp, include_no_drop, include_naive_ensemble)
    else:
        # Original single plot functionality
        rng = np.random.default_rng(seed)
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        
        # Get configs for both discrete and continuous WS
        model_cfg_discrete = get_weak_supervision_cfg(dataset_name, model_size)
        model_cfg_continuous = get_weak_supervision_cfg(dataset_name, model_size)
        model_cfg_continuous['use_continuous'] = True  # Enable continuous mode
        
        # Add tensor decomposition configuration if requested
        if include_tensor_decomp:
            model_cfg_tensor = get_weak_supervision_tensor_cfg(dataset_name, model_size)
            model_cfg_tensor['verifier_names'] = dataset.verifier_names
        
        # Add no-drop configuration if requested
        if include_no_drop:
            model_cfg_no_drop = get_weak_supervision_no_drop_cfg(dataset_name, model_size)
            model_cfg_no_drop['verifier_names'] = dataset.verifier_names
        
        model_cfg_discrete['verifier_names'] = dataset.verifier_names
        model_cfg_continuous['verifier_names'] = dataset.verifier_names
        
        (x_train, y_train) = dataset.test_data
        train_answers = dataset.test_answers

        num_problems, num_responses, num_verifiers = x_train.shape

        # Calculate k_values using pure powers of 2 up to 1024
        k_values = [2**i for i in range(0, 11)]  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
        # Remove any values that exceed num_responses
        k_values = [k for k in k_values if k <= num_responses]
        if num_responses < 1024:
            k_values.append(num_responses)
        k_values = sorted(list(set(k_values))) # Ensure uniqueness and sort, just in case

        majority_topk_values = [1]  # Only show Majority1@k

        # Bootstrap
        B = num_bootstrap_samples
        results_majority = {m: np.zeros((B, len(k_values))) for m in majority_topk_values}
        results_lr = np.zeros((B, len(k_values)))
        results_lr_per_problem = np.zeros((B, len(k_values)))
        results_best_verifier = np.zeros((B, len(k_values)))
        results_pass_at_k = np.zeros((B, len(k_values)))
        results_weak_supervision = np.zeros((B, len(k_values)))
        results_weak_supervision_continuous = np.zeros((B, len(k_values)))

        if include_tensor_decomp:
            results_weak_supervision_tensor = np.zeros((B, len(k_values)))
        if include_no_drop:
            results_weak_supervision_no_drop = np.zeros((B, len(k_values)))
        if include_naive_ensemble:
            results_naive_ensemble = np.zeros((B, len(k_values)))

        for k_idx, k in enumerate(k_values):
            # bootstrap:
            for b in range(B):
                # For k > num_responses, use all available samples
                if k > num_responses:
                    k_actual = num_responses
                else:
                    k_actual = k
                    
                x_sampled = np.zeros((num_problems, k_actual, num_verifiers))
                y_sampled = np.zeros((num_problems, k_actual))
                answers_sampled = []

                for i in range(num_problems):
                    idx = rng.choice(num_responses, size=k_actual, replace=False)
                    x_sampled[i] = x_train[i, idx]
                    y_sampled[i] = y_train[i, idx]
                    answers_sampled.append([train_answers[i][j] for j in idx])

                # Pass@K biased
                #pass_at_k_results = calculate_pass_k_gt(y_sampled, k_values)
                #pass_at_k_results = [pass_at_k_results[k] for k in k_values]
                #results_pass_at_k[b, k_idx] = pass_at_k_results[k]

                #Majority@k
                for m in majority_topk_values:
                    acc, _ = calculate_majority_M_at_k(
                        answers_sampled, y_sampled, k, topM=m, return_mean=True
                    )
                    results_majority[m][b, k_idx] = acc

                # OracleLR (per-data)
                results_lr[b, k_idx] = vanilla_lr([k], x_sampled, y_sampled, model_cfg_discrete['deps_data_fraction'])[0]

                # Calculate naive ensemble accuracy if requested
                if include_naive_ensemble:
                    results_naive_ensemble[b, k_idx] = get_naive_ensemble_accuracy([k], x_sampled, y_sampled)[0]

                # OracleLR (per-problem)
                #results_lr_per_problem[b, k_idx] = vanilla_lr_per_problem([k], x_sampled, y_sampled)[0]

                # Best verifier
                results_best_verifier[b, k_idx] = get_best_verifier(x_sampled, y_sampled)
        
                # Pass@K unbiased
                if k_actual == k:  # Only calculate if we have enough samples
                    results_pass_at_k[b, k_idx] = calculate_pass_k_gt(y_sampled, [k])[k]
                else:  # If we don't have enough samples, use all available samples
                    results_pass_at_k[b, k_idx] = calculate_pass_k_gt(y_sampled, [k_actual])[k_actual]

                # Weak supervision - Discrete
                results_weak_supervision[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_discrete)[0]
                
                # Weak supervision - Continuous
                results_weak_supervision_continuous[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_continuous)[0]

                # Weak supervision - Discrete with Tensor Decomposition
                if include_tensor_decomp:
                    results_weak_supervision_tensor[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_tensor)[0]

                # Weak supervision - Discrete - No Verifier Dropping
                if include_no_drop:
                    results_weak_supervision_no_drop[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_no_drop)[0]
        
        # Pass@K biased
        #pass_at_k_results = calculate_pass_k_gt(y_train, k_values)
        pass_at_k_results = np.mean(results_pass_at_k, axis=0)
        # Let's fix this a bit 
        #print(k_values, pass_at_k_results)
        
        # Calculate unbiased pass@k using actual number of samples available
        k_values_actual = [min(k, num_responses) for k in k_values]
        pass_at_k_results_unbiased = calculate_pass_k_unbiased(y_train, k_values_actual, return_mean=True)
        pass_at_k_results_unbiased = [pass_at_k_results_unbiased[k] for k in k_values_actual]

        #print(k_values, pass_at_k_results_unbiased)
        # --- Plot results
        plt.figure(figsize=(12, 6))
        plt.rcParams.update({'font.size': 14})  # Set global font size
        
        # Set log scale with base 2
        plt.xscale('log', base=2)
        
        # Plot all series with consistent styling
        plt.plot(k_values, pass_at_k_results, 'o--', label="Pass@k (Oracle Verification)", color='red', linewidth=2)

        # Add majority@k
        for m in majority_topk_values:
            mean = np.mean(results_majority[m], axis=0)
            err = np.std(results_majority[m], axis=0)
            plt.plot(k_values, mean, 'o--', label="Majority1@K", color='purple')
            plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

        # Add Supervision 
        mean = np.mean(results_lr, axis=0)
        err = np.std(results_lr, axis=0)
        plt.plot(k_values, mean, 'o--', label="Supervised LR", color='black')  # Bright green
        plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

        # Add Weak Supervision - Discrete
        mean = np.mean(results_weak_supervision, axis=0)
        err = np.std(results_weak_supervision, axis=0)
        plt.plot(k_values, mean, 'o--', label="Weak Supervision - Discrete", color='#2ecc71')  # Bright green
        plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

        # Add Weak Supervision - Continuous
        mean_continuous = np.mean(results_weak_supervision_continuous, axis=0)
        err_continuous = np.std(results_weak_supervision_continuous, axis=0)
        plt.plot(k_values, mean_continuous, 'o--', label="Weak Supervision - Continuous", color='#27ae60')  # Darker green
        plt.fill_between(k_values, mean_continuous - err_continuous, mean_continuous + err_continuous, alpha=0.2)

        # Add Weak Supervision - Discrete with Tensor Decomposition
        if include_tensor_decomp:
            mean_tensor = np.mean(results_weak_supervision_tensor, axis=0)
            err_tensor = np.std(results_weak_supervision_tensor, axis=0)
            plt.plot(k_values, mean_tensor, 'o--', label="Weak Supervision - Discrete + Tensor Decomp", color='#16a085')  # Teal color
            plt.fill_between(k_values, mean_tensor - err_tensor, mean_tensor + err_tensor, alpha=0.2)

        # Add Weak Supervision - Discrete - No Verifier Dropping
        if include_no_drop:
            mean_no_drop = np.mean(results_weak_supervision_no_drop, axis=0)
            err_no_drop = np.std(results_weak_supervision_no_drop, axis=0)
            plt.plot(k_values, mean_no_drop, 'o--', label="Weak Supervision - Discrete - No Verifier Dropping", color='#8e44ad')  # Purple color
            plt.fill_between(k_values, mean_no_drop - err_no_drop, mean_no_drop + err_no_drop, alpha=0.2)

        # Add Naive Ensemble if requested
        if include_naive_ensemble:
            mean = np.mean(results_naive_ensemble, axis=0)
            err = np.std(results_naive_ensemble, axis=0)
            plt.plot(k_values, mean, 'o--', label="Naive Ensemble", color='orange')
            plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

        # Add o3-mini baseline for 70B models
        if model_size == "70B" and dataset_name in O3_MINI_BASELINES:
            baseline = float(O3_MINI_BASELINES[dataset_name] / 100.0)  # Convert to scalar
            plt.axhline(y=baseline, color='#404040', linestyle='--')  # Dark grey, removed label
            plt.text(k_values[0], baseline, f'o3-mini ({baseline*100:.1f}%)', 
                    ha='left', va='bottom', color='#404040')  # Left side

        # Add Llama70B MV baseline for 8B models
        if model_size == "8B" and dataset_name in LLAMA70B_MV_BASELINES:
            baseline = float(LLAMA70B_MV_BASELINES[dataset_name] / 100.0)  # Convert to scalar
            plt.axhline(y=baseline, color='#404040', linestyle='--')  # Dark grey, removed label
            plt.text(k_values[0], baseline, f'Llama70B First Sample ({baseline*100:.1f}%)', 
                    ha='left', va='bottom', color='#404040')  # Left side

        # Add First Sample baseline - use first data point from any series
        first_sample_baseline = float(pass_at_k_results[0])  # Convert to scalar
        plt.axhline(y=first_sample_baseline, color='#A0A0A0', linestyle='--')  # Light grey, removed label
        plt.text(k_values[-1], first_sample_baseline, f'First Sample ({first_sample_baseline*100:.1f}%)', 
                ha='right', va='bottom', color='#A0A0A0')  # Right side

        # Plot coverage
        plt.xlabel('Number of Repeated Generations', fontsize=16, fontweight='bold')
        plt.ylabel('Selection@1 (%)', fontsize=16, fontweight='bold')
        plt.title(f'{get_clean_dataset_name(dataset_name)} - {model_size} Models', fontsize=24, fontweight='bold', pad=10)
        
        # Set x-axis ticks and labels
        plt.xticks(k_values, [f'$2^{{{int(np.log2(k))}}}$' for k in k_values], fontsize=14)
        plt.yticks(fontsize=14)
        
        # Add grid
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        
        # Create custom legend with specified order
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Define the desired order of approaches
        desired_order = [
            'Pass@k (Oracle Verification)',
            'Supervised LR',
            'Weak Supervision - Discrete',
            'Weak Supervision - Continuous',
        ]
        
        if include_tensor_decomp:
            desired_order.append('Weak Supervision - Discrete + Tensor Decomp')
        if include_no_drop:
            desired_order.append('Weak Supervision - Discrete - No Verifier Dropping')
        if include_naive_ensemble:
            desired_order.append('Naive Ensemble')
            
        desired_order.append('Majority1@K')
        
        # Reorder handles and labels according to desired order
        ordered_handles = []
        ordered_labels = []
        for label in desired_order:
            if label in labels:
                idx = labels.index(label)
                ordered_handles.append(handles[idx])
                ordered_labels.append(labels[idx])
        
        # Create single legend
        plt.legend(ordered_handles, ordered_labels,
                  bbox_to_anchor=(1.05, 0.5),  # Centered vertically
                  loc='center left',
                  borderaxespad=0.,
                  fontsize=14)
                
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / f"coverage_{dataset_name}_{model_size}_verifier{verifier_size}.png", bbox_inches='tight')
        plt.close()

        # Create and print results table
        print(f"\nResults for {dataset_name} ({model_size}):")
        print("-" * 100)
        
        # Create table header
        header = "Approach".ljust(40)
        for k in k_values:
            header += f"k={k}".rjust(10)
        print(header)
        print("-" * 100)
        
        # Add each approach's results
        approaches = []
        results = []
        
        # Pass@k
        approaches.append("Pass@k (Oracle Verification)")
        results.append(pass_at_k_results)
        
        # Supervised LR
        approaches.append("Supervised LR")
        results.append(np.mean(results_lr, axis=0))
        
        # Weak Supervision - Discrete
        approaches.append("Weak Supervision - Discrete")
        results.append(np.mean(results_weak_supervision, axis=0))
        
        # Weak Supervision - Continuous
        approaches.append("Weak Supervision - Continuous")
        results.append(np.mean(results_weak_supervision_continuous, axis=0))
        
        # Weak Supervision - Discrete + Tensor Decomp
        if include_tensor_decomp:
            approaches.append("Weak Supervision - Discrete + Tensor Decomp")
            results.append(np.mean(results_weak_supervision_tensor, axis=0))
        
        # Weak Supervision - Discrete - No Verifier Dropping
        if include_no_drop:
            approaches.append("Weak Supervision - Discrete - No Verifier Dropping")
            results.append(np.mean(results_weak_supervision_no_drop, axis=0))
        
        # Naive Ensemble
        if include_naive_ensemble:
            approaches.append("Naive Ensemble")
            results.append(np.mean(results_naive_ensemble, axis=0))
        
        # Majority1@K
        approaches.append("Majority1@K")
        results.append(np.mean(results_majority[1], axis=0))
        
        # Print each row
        for approach, result in zip(approaches, results):
            row = approach.ljust(40)
            for val in result:
                row += f"{val*100:10.1f}"
            print(row)
        
        print("-" * 100)
        print()
    
    # Add power law fits:
    dataset_results = {
        'Pass@k (Oracle Verification)': pass_at_k_results,
        'Weak Supervision - Discrete': np.mean(results_weak_supervision, axis=0),
        'Weak Supervision - Continuous': np.mean(results_weak_supervision_continuous, axis=0),
        'Majority1@K': np.mean(results_majority[1], axis=0),
    }
    if include_naive_ensemble:
        dataset_results['Naive Ensemble'] = np.mean(results_naive_ensemble, axis=0)

    desired_order = ['Pass@k (Oracle Verification)',
                     'Majority1@K',
                     'Naive Ensemble',
                     'Weak Supervision - Continuous',
                     'Weak Supervision - Discrete',
    ]

    if len(k_values) > 3:
        add_fitted_power_law({dataset_name: dataset_results}, desired_order, k_values, dataset_name, model_size)


    # Add verifier generation vs ensemble size:
    # Get verifier scores:
    """
   
    # Plotting
    verifier_f1_scores = get_verifier_scores(x_train, y_train)
    # sort from worst to best
    verifier_indices = np.argsort(verifier_f1_scores)#[::-1]

    accuracy_matrix, all_responses, all_verifiers = verifier_generation_vs_ensemble_size(x_train, y_train, verifier_indices)
    # Plotting
    fig_name = f"scaling_matrix_from_worst_{dataset_name}_{model_size}_verifier{verifier_size}.png"
    extra='ranked by quality, worst first'
    plot_accuracy_matrix(accuracy_matrix,
                        all_responses,
                        all_verifiers,
                        dataset.dataset_name, dataset.model_size, fig_name, verifier_size, extra=extra)

    fig_name = f"scaling_from_worst_{dataset_name}_{model_size}_verifier{verifier_size}.png"
    make_plot_law(accuracy_matrix, all_responses, all_verifiers, fig_name, dataset_name, model_size, verifier_size, extra=extra)

     # sort from best to worst
    verifier_indices = np.argsort(verifier_f1_scores)[::-1]

    accuracy_matrix, all_responses, all_verifiers = verifier_generation_vs_ensemble_size(x_train, y_train, verifier_indices)
    # Plotting
    fig_name = f"scaling_matrix_from_best_{dataset_name}_{model_size}_verifier{verifier_size}.png"
    extra='ranked by quality, best first'
    plot_accuracy_matrix(accuracy_matrix,
                        all_responses,
                        all_verifiers,
                        dataset.dataset_name, dataset.model_size, fig_name, verifier_size, extra=extra)
    fig_name = f"scaling_from_best_{dataset_name}_{model_size}_verifier{verifier_size}.png"
    make_plot_law(accuracy_matrix, all_responses, all_verifiers, fig_name, dataset_name, model_size, verifier_size, extra=extra)
    """
    return


def get_naive_ensemble_accuracy(k_values, x_train, y_train):
    """
    """
    oracle_lr = []
    for k_ in k_values:
        x_train2 = x_train.copy()
        y_train2 = y_train.copy()

        # Select the first k_ attempts:
        # TODO: THIs is a biased estimator because it assumes samples are iid
        x_train2 = x_train2[:, :k_, :]
        y_train2 = y_train2[:, :k_]

        ensemble_scores = x_train2.mean(-1) # num_problems, num_responses

        # Get the naive ensemble accuracy

        # Get the verifier with the highest selection accuracy
        top_index = np.argmax(ensemble_scores, 1) #num_problems, 

        pred_labels = y_train2[np.arange(len(top_index)), top_index] # pred_labels

        # Step 3: Check if the top-selected response is correct
        select_acc = np.mean(pred_labels)

        oracle_lr.append(select_acc)

    return oracle_lr


def vanilla_lr(k_values, x_train, y_train, split=0.8):
    oracle_lr = []

    for k_ in k_values:
        x_train2 = x_train.copy()
        y_train2 = y_train.copy()

        # Select the first k_ attempts:
        # TODO: THIs is a biased estimator because it assumes samples are iid
        x_train2 = x_train2[:, :k_, :]
        y_train2 = y_train2[:, :k_]

        num_queries, num_responses, num_verifiers = x_train2.shape

        # Flatten the data
        X = x_train2.reshape((num_queries*num_responses, num_verifiers)) *2 -1
        y = np.array(y_train2).reshape((num_queries*num_responses,))

        num_samples = max(1, int(num_queries*num_responses*split))

        X_train3 = X[:num_samples]
        y_train3 = y[:num_samples]

        # Check if we have at least 2 classes
        if len(np.unique(y_train3)) < 2:
            # If only one class, return the majority class accuracy
            majority_class = np.unique(y_train3)[0]
            select_acc = np.mean(y == majority_class)
            oracle_lr.append(select_acc)
            continue

        # Fit a LR model to all the data 
        model = LogisticRegression(class_weight="balanced")
        model.fit(X_train3, y_train3)

        # Check accuracy:
        # 1. Predict class labels and probabilities
        y_pred = model.predict(X) # (num_queries*num_responses, )
        y_proba = model.predict_proba(X)[:, 1]  # probability for class 1
        #print('Accuracy:', accuracy_score(y, y_pred))

        # Select sample with the highest score for each answer
        y_pred2 = y_pred.reshape((num_queries, num_responses))
        y_proba2 = y_proba.reshape((num_queries, num_responses))

        # Select index of top candidate by predicted probability
        top_index = np.argmax(y_proba2, 1) #(num_queries, )


        # is the top-ranked candidate according to the model correct?        
        pred_labels = y_train2[np.arange(len(top_index)), top_index] # pred_labels


        # whether it is possible to get a correct answer in top-k
        # true_labels = np.any(y_train2, 1).astype(int)

        # Number of examples where the top-k answer is correct:
        select_acc = np.mean(pred_labels)

        oracle_lr.append(select_acc)

    return oracle_lr


def vanilla_weak_supervision(k_values, x_train, y_train, model_cfg):
    oracle_lr = []
    for k_ in k_values:
        x_train2 = x_train.copy()
        y_train2 = y_train.copy()

        # Select the first k_ attempts:
        # TODO: THIs is a biased estimator because it assumes samples are iid
        x_train2 = x_train2[:, :k_, :]
        y_train2 = y_train2[:, :k_]

        num_queries, num_responses, num_verifiers = x_train2.shape

        # Flatten the data
        X = x_train2.reshape((num_queries*num_responses, num_verifiers))
        y = np.array(y_train2).reshape((num_queries*num_responses,))

        if not(model_cfg['use_continuous']):
            reward_threshold = model_cfg["reward_threshold"]
            X = (X >= reward_threshold).astype(int)
        # Fit a LR model to all the data 
        model = WeakSupervised(**model_cfg)

        model.fit(X, y)
        # 1. Predict class labels and probabilities
        #X = X[:, model.verifier_idxs]
        y_proba = model.predict_proba(X + 1)[:,1] # (num_queries*num_responses, 2)
        #verifier_ranking = model.get_verifier_ranking(X, y)
        # verifier ranking:
        #selected_indices = model.verifier_idxs
        #verifier_ranking = selected_indices[verifier_ranking]
        # Each problem has a selected
        # Select the top-ranked candidate:

        # Select sample with the highest score for each answer
        y_proba2 = y_proba.reshape((num_queries, num_responses))

        # Select index of top candidate by predicted probability
        top_index = np.argmax(y_proba2, 1) #(num_queries, )

        # is the top-ranked candidate according to the model correct?        
        pred_labels = y_train2[np.arange(len(top_index)), top_index] # pred_labels

        # whether it is possible to get a correct answer in top-k
        # true_labels = np.any(y_train2, 1).astype(int)

        # Number of examples where the top-k answer is correct:
        select_acc = np.mean(pred_labels)

        oracle_lr.append(select_acc)

    return oracle_lr


def vanilla_lr_per_problem(k_values, x_train, y_train):
    """
    Trains a separate LR model per problem, using the first k responses for each problem.
    Returns selection accuracy per k (averaged over all problems).
    """
    num_problems, num_total_responses, num_verifiers = x_train.shape
    oracle_lr = []

    for k_ in k_values:
        correct = 0
        total = num_problems    # include all problems

        for i in range(num_problems):
            if k_ > num_total_responses:
                continue  # skip if not enough responses

            X = x_train[i, :k_, :]  # shape (k_, num_verifiers)
            y = y_train[i, :k_]     # shape (k_,)

            # Skip problems with only one class label
            #if np.all(y == 0) or np.all(y == 1):
            #    # Need to use a different model for this problem!
            #    
            # continue
             # Case 1: All responses are incorrect
            if np.all(y == 0):
                # Choose response with highest average verifier score
                avg_scores = X.mean(axis=1)
                top_idx = np.argmax(avg_scores)
                # This will always be wrong (y == 0), so no increment
                continue

            # Case 2: All responses are correct
            elif np.all(y == 1):
                # Any selection is correct
                correct += 1
                continue

            # Case 3: Mixed correct/incorrect responses
            else:
                model = LogisticRegression(class_weight="balanced")
                model.fit(X, y)

                probs = model.predict_proba(X)[:, 1]
                top_idx = np.argmax(probs)
                if y[top_idx] == 1:
                    correct += 1
                #pass  # failure → count as incorrect (do not increment correct)

        acc = correct / total if total > 0 else 0.0
        oracle_lr.append(acc)

    return oracle_lr


def get_best_verifier(x_train, y_train):
    """
    """
    # Get the highest scoring sample for each problem and verifier
    top_response_indices = np.argmax(x_train, 1) #num_problems, num_verifier

    # Step 3: Check if the top-selected response is correct
    selected_correctness = np.stack([y_train[i, idx] for i, idx in enumerate(top_response_indices)]) #num_problems, num_verifier

    # Compute selection accuracy:
    selection_accuracy = np.mean(selected_correctness, 0) # (num_verifiers, )

    best_verifier_acc = np.max(selection_accuracy)
    return best_verifier_acc
  

def f1_score_numpy(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def get_selection_accuracy_atk(scores, y_train, k=1, reduce='top'):
    """
    # scores shape: (num_problems, num_responses)
    # k: use first k responses
    
    Need to turn this into an unbiased estimator
    """
    scores = scores[:, :k] # num_problems, num_responses
    y_train = y_train[:, :k]

    # Step 2: Select top-scoring respons e for each problem
    if reduce == 'top':        
        top_response_indices = scores.argmax(axis=1)  # shape: (num_problems,)
        # Step 3: Check if the top-selected response is correct
        selected_correctness = [y_train[i, idx] for i, idx in enumerate(top_response_indices)]
    else:
        pass
    assert len(selected_correctness) == scores.shape[0]
    # Compute selection accuracy:
    selection_accuracy_best = np.mean(selected_correctness)
    return selection_accuracy_best


def get_verifier_scores(x_train, y_train):
    # Number of generations vs number of verifies in ensemble
    num_queries, num_responses, num_verifiers = x_train.shape
    # How do we rank verifiers vs samples
    verifier_top_pred_indices = np.argmax(x_train, axis=1) # num_problems, num_Verifiers
    verifier_top_pred_correctness = np.stack([y_train[i, idx] for i, idx in enumerate(verifier_top_pred_indices)])
    tp = np.sum(verifier_top_pred_correctness, axis=0)
    fp = np.sum(1 - verifier_top_pred_correctness, axis=0)
    fn = np.sum(verifier_top_pred_correctness, axis=0)
    tpr = tp / (tp + fn + 1e-8)
    tnr = fp / (fp + tp + 1e-8)
    balanced_accuracy = 0.5 * tpr * tnr
    
    return balanced_accuracy


def verifier_generation_vs_ensemble_size(x_train, y_train, verifier_indices):
    """
    """
    # ----
    num_queries, num_responses, num_verifiers = x_train.shape

    # ---- 
    all_responses = np.linspace(10, num_responses, 8).astype(int)
    all_verifiers = np.linspace(1, num_verifiers, 10).astype(int)

    # Initialize accuracy matrix: (num_responses, num_verifiers)
    accuracy_matrix = np.zeros((len(all_responses), len(all_verifiers)))

    # For each response (generation) and verifier, compute accuracy
    for r_idx, r in enumerate(all_responses):
        for v_idx, v in enumerate(all_verifiers):
            # get scores for selected verifiers
            all_v = verifier_indices[:v]
            scores = x_train[..., all_v].mean(-1)
            assert scores.shape == (num_queries, num_responses)
            # get selection accuracy score
            acc = get_selection_accuracy_atk(scores, y_train, r)

            accuracy_matrix[r_idx, v_idx] = acc

    return accuracy_matrix, all_responses, all_verifiers


def power_law(x, a, b, c):
    return a * (x ** b) + c


# R-squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def make_plot_law(accuracy_matrix, all_responses, all_verifiers, fig_name, dataset_name, model_size, verifier_size, extra=""):
    # Set up figure
    plt.figure(figsize=(10, 6))

    Z = accuracy_matrix.T
    cmap = cm.get_cmap("plasma")  # You can change to "plasma", "coolwarm", etc.
    norm = mcolors.Normalize(vmin=0, vmax=len(all_verifiers) - 1)

    #colors = [f'C{idx}' for idx in range(15)]
    # Fit curve for each verifier level
    for idx, v in enumerate(all_verifiers):
        x = all_responses
        y = Z[idx]  # accuracy across generations for verifier level `v`
        color = cmap(norm(idx))  # gradient color

        try:
            params, _ = curve_fit(power_law, x, y, maxfev=10000)
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = power_law(x_fit, *params)

            plt.plot(x, y, 'o', color=color)
            plt.plot(x_fit, y_fit, '-', label=f'Ensemble Size={v}', color=color)
        except RuntimeError:
            print(f"Curve fitting failed for verifier={v}")
            continue

    plt.xlabel("Number of Generations")
    plt.ylabel("Selection Accuracy")
    plt.title(f"Effect of Naive Ensemble Size: {dataset_name} (Model {model_size}) (Verifier <={verifier_size}) {extra}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / fig_name)
    plt.close()


def plot_accuracy_matrix(accuracy_matrix, all_responses, all_verifiers, dataset_name, model_size, fig_name, verifier_size, extra=""):
    # Plotting
    plt.figure(figsize=(10, 6))
    im = plt.imshow(accuracy_matrix.T, aspect='auto', cmap='plasma', origin='lower')#, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='SelectionAccuracy')
    plt.xlabel('Number of Generations')
    plt.xticks(np.arange(len(all_responses)), labels=all_responses)
    plt.ylabel(f"Number of Verifiers in Ensemble {extra}")
    plt.yticks(np.arange(len(all_verifiers)), labels=all_verifiers)
    plt.title(f"Selection Accuracy for {dataset_name} {model_size} (Verifier Size: {verifier_size})")

    # Add text annotations
    for i in range(len(all_responses)):
        for j in range(len(all_verifiers)):
            plt.text(i, j, f"{accuracy_matrix[i, j]:.2f}", 
                    ha='center', va='center', color='white' if accuracy_matrix[i, j] < 0.6 else 'black')

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / fig_name)
    plt.close()
    return


def debug_plot(dataset_name, model_size, verifier_size):
    """
    Debug mode that generates dummy data and plots it in the same format as the main function.
    This is useful for testing visualization formatting without loading real data.
    """
    # Generate dummy k values
    k_values = [2**i for i in range(0, 8)]  # 1, 2, 4, 8, 16, 32, 64, 128
    
    # Generate dummy data for each series
    # We'll create some realistic-looking curves with some noise
    def generate_dummy_curve(base_value, growth_rate, noise_scale=0.02):
        curve = base_value * (1 + growth_rate * np.log2(np.array(k_values)))
        noise = np.random.normal(0, noise_scale, len(k_values))
        return curve + noise

    # Generate dummy results for each method
    results_pass_at_k = generate_dummy_curve(0.3, 0.1)
    results_majority = {1: generate_dummy_curve(0.4, 0.08)}
    results_lr = generate_dummy_curve(0.5, 0.12)
    results_weak_supervision = generate_dummy_curve(0.6, 0.15)
    results_weak_supervision_continuous = generate_dummy_curve(0.65, 0.16)  # Slightly higher base value and growth rate
    results_best_verifier = generate_dummy_curve(0.45, 0.09)
    results_naive_ensemble = generate_dummy_curve(0.55, 0.11)

    # --- Plot results
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 14})  # Set global font size
    
    # Set log scale with base 2
    plt.xscale('log', base=2)
    
    # Plot all series with consistent styling
    plt.plot(k_values, results_pass_at_k, 'o--', label="Pass@k (Oracle Verification)", color='red', linewidth=2)

    # Add majority@k
    for m in results_majority:
        plt.plot(k_values, results_majority[m], 'o--', label=f"Majority{m}@k", color='purple')
        # Add some dummy error bars
        err = 0.02 * np.ones_like(k_values)
        plt.fill_between(k_values, results_majority[m] - err, results_majority[m] + err, alpha=0.2)

    # Add OracleLR (per data)
    plt.plot(k_values, results_lr, 'o--', label="OracleLR (per data)", color='blue')
    err = 0.02 * np.ones_like(k_values)
    plt.fill_between(k_values, results_lr - err, results_lr + err, alpha=0.2)

    # Add Weak Supervision - Discrete
    plt.plot(k_values, results_weak_supervision, 'o--', label="Weak Supervision - Discrete", color='#2ecc71')
    err = 0.02 * np.ones_like(k_values)
    plt.fill_between(k_values, results_weak_supervision - err, results_weak_supervision + err, alpha=0.2)

    # Add Weak Supervision - Continuous
    plt.plot(k_values, results_weak_supervision_continuous, 'o--', label="Weak Supervision - Continuous", color='#27ae60')
    err_continuous = 0.02 * np.ones_like(k_values)
    plt.fill_between(k_values, results_weak_supervision_continuous - err_continuous, results_weak_supervision_continuous + err_continuous, alpha=0.2)

    # Add Best Verifier
    plt.plot(k_values, results_best_verifier, 'o--', label="Best Verifier", color='blue')
    err = 0.02 * np.ones_like(k_values)
    plt.fill_between(k_values, results_best_verifier - err, results_best_verifier + err, alpha=0.2)

    # Add Naive Ensemble
    plt.plot(k_values, results_naive_ensemble, 'o--', label="Naive Ensemble", color='orange')
    err = 0.02 * np.ones_like(k_values)
    plt.fill_between(k_values, results_naive_ensemble - err, results_naive_ensemble + err, alpha=0.2)

    # Add o3-mini baseline for 70B models
    if model_size == "70B" and dataset_name in O3_MINI_BASELINES:
        baseline = O3_MINI_BASELINES[dataset_name] / 100.0
        plt.axhline(y=baseline, color='#404040', linestyle='--')  # Dark grey, removed label
        plt.text(k_values[0], baseline, f'o3-mini ({baseline*100:.1f}%)', 
                ha='left', va='bottom', color='#404040')  # Left side

    # Add Llama70B MV baseline for 8B models
    if model_size == "8B" and dataset_name in LLAMA70B_MV_BASELINES:
        baseline = LLAMA70B_MV_BASELINES[dataset_name] / 100.0
        plt.axhline(y=baseline, color='#404040', linestyle='--')  # Dark grey, removed label
        plt.text(k_values[0], baseline, f'Llama70B First Sample ({baseline*100:.1f}%)', 
                ha='left', va='bottom', color='#404040')  # Left side

    # Add First Sample baseline - use first data point from any series
    first_sample_baseline = results_pass_at_k[0]  # Use first point from pass@k series
    plt.axhline(y=first_sample_baseline, color='#A0A0A0', linestyle='--')  # Light grey, removed label
    plt.text(k_values[-1], first_sample_baseline, f'First Sample ({first_sample_baseline*100:.1f}%)', 
            ha='right', va='bottom', color='#A0A0A0')  # Right side

    # Plot coverage
    plt.xlabel('Number of Repeated Generations', fontsize=16, fontweight='bold')
    plt.ylabel('Selection@1 (%)', fontsize=16, fontweight='bold')
    plt.title(f'{dataset_name} - {model_size} Models', fontsize=24, fontweight='bold', pad=10)
    
    # Set x-axis ticks and labels
    plt.xticks(k_values, [f'$2^{{{int(np.log2(k))}}}$' for k in k_values], fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    
    # Create custom legend with specified order
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Define the desired order of approaches
    desired_order = [
        'Pass@k (Oracle Verification)',
        'Weak Supervision - Discrete',
        'Weak Supervision - Continuous',
        'Naive Ensemble',
        'Majority1@K'
    ]
    
    # Reorder handles and labels according to desired order
    ordered_handles = []
    ordered_labels = []
    for label in desired_order:
        if label in labels:
            idx = labels.index(label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    
    # Create single legend
    plt.legend(ordered_handles, ordered_labels,
              bbox_to_anchor=(1.05, 0.5),  # Centered vertically
              loc='center left',
              borderaxespad=0.,
              fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"coverage_{dataset_name}_{model_size}_verifier{verifier_size}_debug.png", bbox_inches='tight')
    plt.close()

    # Add plot so that we can see the verifier generation vs ensemble size: 
    # how to rank


def plot_grid(datasets, model_size, verifier_size, num_bootstrap_samples=1, seed=0, include_tensor_decomp=False, include_no_drop=False, include_naive_ensemble=False):
    """
    Create a 2x3 grid of plots for multiple datasets.
    """
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()
    
    # Define the desired order of approaches for the legend
    desired_order = [
        'Pass@k (Oracle Verification)',
        'Weak Supervision - Discrete',
        'Weak Supervision - Continuous',
    ]
    
    if include_tensor_decomp:
        desired_order.append('Weak Supervision - Discrete + Tensor Decomp')
    if include_no_drop:
        desired_order.append('Weak Supervision - Discrete - No Verifier Dropping')
    if include_naive_ensemble:
        desired_order.append('Naive Ensemble')
        
    desired_order.append('Majority1@K')
    
    # Define the specific order of datasets for the grid
    grid_order = [
        'AIMO-v2', 'MATH-500-v2', 'GPQA-v2-Diamond',  # Top row
        'MMLU-College-v2', 'MMLU-Pro-v2', 'BBH-v2'    # Bottom row
    ]
    
    # Store all handles and labels for the shared legend
    all_handles = []
    all_labels = []
    
    # Store results for all datasets
    all_dataset_results = {}
    
    for idx, dataset_name in enumerate(grid_order):
        if idx >= n_rows * n_cols:
            break
            
        ax = axes[idx]
        
        # Get data and results for this dataset
        rng = np.random.default_rng(seed)
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        
        # Get configs for both discrete and continuous WS
        model_cfg_discrete = get_weak_supervision_cfg(dataset_name, model_size)
        model_cfg_continuous = get_weak_supervision_cfg(dataset_name, model_size)
        model_cfg_continuous['use_continuous'] = True
        
        # Add tensor decomposition configuration if requested
        if include_tensor_decomp:
            model_cfg_tensor = get_weak_supervision_tensor_cfg(dataset_name, model_size)
            model_cfg_tensor['verifier_names'] = dataset.verifier_names
        
        # Add no-drop configuration if requested
        if include_no_drop:
            model_cfg_no_drop = get_weak_supervision_no_drop_cfg(dataset_name, model_size)
            model_cfg_no_drop['verifier_names'] = dataset.verifier_names
        
        model_cfg_discrete['verifier_names'] = dataset.verifier_names
        model_cfg_continuous['verifier_names'] = dataset.verifier_names
        
        (x_train, y_train) = dataset.test_data
        train_answers = dataset.test_answers
        
        num_problems, num_responses, num_verifiers = x_train.shape
        k_values = [2**i for i in range(0, 8)]
        majority_topk_values = [1]
        
        # Bootstrap
        B = num_bootstrap_samples
        results_majority = {m: np.zeros((B, len(k_values))) for m in majority_topk_values}
        results_lr = np.zeros((B, len(k_values)))
        results_lr_per_problem = np.zeros((B, len(k_values)))
        results_best_verifier = np.zeros((B, len(k_values)))
        results_pass_at_k = np.zeros((B, len(k_values)))
        results_weak_supervision = np.zeros((B, len(k_values)))
        results_weak_supervision_continuous = np.zeros((B, len(k_values)))
        if include_tensor_decomp:
            results_weak_supervision_tensor = np.zeros((B, len(k_values)))
        if include_no_drop:
            results_weak_supervision_no_drop = np.zeros((B, len(k_values)))
        if include_naive_ensemble:
            results_naive_ensemble = np.zeros((B, len(k_values)))

        for k_idx, k in enumerate(k_values):
            # bootstrap:
            for b in range(B):
                # For k=128, use all available samples (100)
                if k > num_responses:
                    k_actual = num_responses
                else:
                    k_actual = k
                    
                x_sampled = np.zeros((num_problems, k_actual, num_verifiers))
                y_sampled = np.zeros((num_problems, k_actual))
                answers_sampled = []

                for i in range(num_problems):
                    idx = rng.choice(num_responses, size=k_actual, replace=False)
                    x_sampled[i] = x_train[i, idx]
                    y_sampled[i] = y_train[i, idx]
                    answers_sampled.append([train_answers[i][j] for j in idx])

                # Pass@K biased
                #pass_at_k_results = calculate_pass_k_gt(y_sampled, k_values)
                #pass_at_k_results = [pass_at_k_results[k] for k in k_values]
                #results_pass_at_k[b, k_idx] = pass_at_k_results[k]

                #Majority@k
                for m in majority_topk_values:
                    acc, _ = calculate_majority_M_at_k(
                        answers_sampled, y_sampled, k, topM=m, return_mean=True
                    )
                    results_majority[m][b, k_idx] = acc

                # OracleLR (per-data)
                results_lr[b, k_idx] = vanilla_lr([k], x_sampled, y_sampled)[0]

                # Calculate naive ensemble accuracy if requested
                if include_naive_ensemble:
                    results_naive_ensemble[b, k_idx] = get_naive_ensemble_accuracy([k], x_sampled, y_sampled)[0]

                # OracleLR (per-problem)
                #results_lr_per_problem[b, k_idx] = vanilla_lr_per_problem([k], x_sampled, y_sampled)[0]

                # Best verifier
                results_best_verifier[b, k_idx] = get_best_verifier(x_sampled, y_sampled)
        
                # Pass@K unbiased
                if k_actual == k:  # Only calculate if we have enough samples
                    results_pass_at_k[b, k_idx] = calculate_pass_k_gt(y_sampled, [k])[k]
                else:  # If we don't have enough samples, use all available samples
                    results_pass_at_k[b, k_idx] = calculate_pass_k_gt(y_sampled, [k_actual])[k_actual]

                # Weak supervision - Discrete
                results_weak_supervision[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_discrete)[0]
                
                # Weak supervision - Continuous
                results_weak_supervision_continuous[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_continuous)[0]

                # Weak supervision - Discrete with Tensor Decomposition
                if include_tensor_decomp:
                    results_weak_supervision_tensor[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_tensor)[0]

                # Weak supervision - Discrete - No Verifier Dropping
                if include_no_drop:
                    results_weak_supervision_no_drop[b, k_idx] = vanilla_weak_supervision([k_actual], x_sampled, y_sampled, model_cfg_no_drop)[0]
        
        # Store results for this dataset
        dataset_results = {
            'Pass@k (Oracle Verification)': np.mean(results_pass_at_k, axis=0),
            'Weak Supervision - Discrete': np.mean(results_weak_supervision, axis=0),
            'Weak Supervision - Continuous': np.mean(results_weak_supervision_continuous, axis=0),
        }
        
        if include_tensor_decomp:
            dataset_results['Weak Supervision - Discrete + Tensor Decomp'] = np.mean(results_weak_supervision_tensor, axis=0)
        if include_no_drop:
            dataset_results['Weak Supervision - Discrete - No Verifier Dropping'] = np.mean(results_weak_supervision_no_drop, axis=0)
        if include_naive_ensemble:
            dataset_results['Naive Ensemble'] = np.mean(results_naive_ensemble, axis=0)
        
        dataset_results['Majority1@K'] = np.mean(results_majority[1], axis=0)
        
        all_dataset_results[dataset_name] = dataset_results
        
        # Plot on the current subplot
        ax.set_xscale('log', base=2)
        
        # Plot all series with consistent styling
        pass_at_k_results = np.mean(results_pass_at_k, axis=0)
        line1, = ax.plot(k_values, pass_at_k_results, 'o--', color='red', linewidth=2)
        if idx == 0:  # Only add to legend once
            all_handles.append(line1)
            all_labels.append("Pass@k (Oracle Verification)")
        
        # Add majority@k
        for m in majority_topk_values:
            mean = np.mean(results_majority[m], axis=0)
            err = np.std(results_majority[m], axis=0)
            line2, = ax.plot(k_values, mean, 'o--', color='purple')
            ax.fill_between(k_values, mean - err, mean + err, alpha=0.2)
            if idx == 0:  # Only add to legend once
                all_handles.append(line2)
                all_labels.append("Majority1@K")
        
        # Add Weak Supervision - Discrete
        mean = np.mean(results_weak_supervision, axis=0)
        err = np.std(results_weak_supervision, axis=0)
        line3, = ax.plot(k_values, mean, 'o--', color='#2ecc71')
        ax.fill_between(k_values, mean - err, mean + err, alpha=0.2)
        if idx == 0:  # Only add to legend once
            all_handles.append(line3)
            all_labels.append("Weak Supervision - Discrete")
        
        # Add Weak Supervision - Continuous
        mean_continuous = np.mean(results_weak_supervision_continuous, axis=0)
        err_continuous = np.std(results_weak_supervision_continuous, axis=0)
        line4, = ax.plot(k_values, mean_continuous, 'o--', color='#27ae60')
        ax.fill_between(k_values, mean_continuous - err_continuous, mean_continuous + err_continuous, alpha=0.2)
        if idx == 0:  # Only add to legend once
            all_handles.append(line4)
            all_labels.append("Weak Supervision - Continuous")
        
        # Add Weak Supervision - Discrete with Tensor Decomposition
        if include_tensor_decomp:
            mean_tensor = np.mean(results_weak_supervision_tensor, axis=0)
            err_tensor = np.std(results_weak_supervision_tensor, axis=0)
            line6, = ax.plot(k_values, mean_tensor, 'o--', color='#16a085')
            ax.fill_between(k_values, mean_tensor - err_tensor, mean_tensor + err_tensor, alpha=0.2)
            if idx == 0:  # Only add to legend once
                all_handles.append(line6)
                all_labels.append("Weak Supervision - Discrete + Tensor Decomp")
        
        # Add Weak Supervision - Discrete - No Verifier Dropping
        if include_no_drop:
            mean_no_drop = np.mean(results_weak_supervision_no_drop, axis=0)
            err_no_drop = np.std(results_weak_supervision_no_drop, axis=0)
            line7, = ax.plot(k_values, mean_no_drop, 'o--', color='#8e44ad')
            ax.fill_between(k_values, mean_no_drop - err_no_drop, mean_no_drop + err_no_drop, alpha=0.2)
            if idx == 0:  # Only add to legend once
                all_handles.append(line7)
                all_labels.append("Weak Supervision - Discrete - No Verifier Dropping")
        
        # Add Naive Ensemble if requested
        if include_naive_ensemble:
            mean = np.mean(results_naive_ensemble, axis=0)
            err = np.std(results_naive_ensemble, axis=0)
            line5, = ax.plot(k_values, mean, 'o--', color='orange')
            ax.fill_between(k_values, mean - err, mean + err, alpha=0.2)
            if idx == 0:  # Only add to legend once
                all_handles.append(line5)
                all_labels.append("Naive Ensemble")
        
        # Add First Sample baseline
        first_sample_baseline = float(pass_at_k_results[0])  # Convert to scalar
        ax.axhline(y=first_sample_baseline, color='#A0A0A0', linestyle='--')
        ax.text(k_values[-1], first_sample_baseline, f'First Sample ({first_sample_baseline*100:.1f}%)', 
                ha='right', va='bottom', color='#A0A0A0')
        
        # Add o3-mini baseline for 70B models
        if model_size == "70B" and dataset_name in O3_MINI_BASELINES:
            baseline = float(O3_MINI_BASELINES[dataset_name] / 100.0)  # Convert to scalar
            ax.axhline(y=baseline, color='#404040', linestyle='--')
            ax.text(k_values[0], baseline, f'o3-mini ({baseline*100:.1f}%)', 
                    ha='left', va='bottom', color='#404040')
        
        # Add Llama70B MV baseline for 8B models
        if model_size == "8B" and dataset_name in LLAMA70B_MV_BASELINES:
            baseline = float(LLAMA70B_MV_BASELINES[dataset_name] / 100.0)  # Convert to scalar
            ax.axhline(y=baseline, color='#404040', linestyle='--')
            ax.text(k_values[0], baseline, f'Llama70B First Sample ({baseline*100:.1f}%)', 
                    ha='left', va='bottom', color='#404040')
        
        # Set subplot title and labels
        ax.set_title(f'{get_clean_dataset_name(dataset_name)}', fontsize=14)
        ax.set_xlabel('Number of Repeated Generations', fontsize=12)
        ax.set_ylabel('Selection@1 (%)', fontsize=12)
        ax.set_xticks(k_values)
        ax.set_xticklabels([f'$2^{{{int(np.log2(k))}}}$' for k in k_values], fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    
    # Remove any unused subplots
    for idx in range(len(datasets), n_rows * n_cols):
        fig.delaxes(axes[idx])
    
    # Create a single shared legend
    fig.legend(all_handles, all_labels,
              bbox_to_anchor=(1.0, 0.5),  # Position legend on the right side
              loc='center left',
              borderaxespad=0.,
              fontsize=12)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on the right for the legend
    
    # Save the figure
    plt.savefig(FIGURE_PATH / f"coverage_grid_{model_size}_verifier{verifier_size}.png", bbox_inches='tight')
    plt.close()

    # Print results tables for all datasets
    print(f"\nResults for {model_size} models:")
    print("=" * 100)
    
    for dataset_name in grid_order:
        if dataset_name not in all_dataset_results:
            continue
            
        print(f"\n{dataset_name}:")
        print("-" * 100)
        
        # Create table header
        header = "Approach".ljust(40)
        for k in k_values:
            header += f"k={k}".rjust(10)
        print(header)
        print("-" * 100)
        
        # Print each approach's results
        for approach in desired_order:
            if approach in all_dataset_results[dataset_name]:
                row = approach.ljust(40)
                for val in all_dataset_results[dataset_name][approach]:
                    row += f"{val*100:10.1f}"
                print(row)
        
        print("-" * 100)
    
    print("\n" + "=" * 100)

    add_fitted_power_law(all_dataset_results, desired_order, k_values, dataset_name, model_size)
    return


def add_fitted_power_law(all_dataset_results, desired_order, k_values, dataset_name, model_size):

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', 'D', '^', 'P', 'X', 'd', 'H', 'v', '>', '<']
    fit_results = []

    for i, approach in enumerate(desired_order):
        if approach not in all_dataset_results[dataset_name]:
            continue

        y_vals = np.array(all_dataset_results[dataset_name][approach]) #* 100
        x_vals = np.array(k_values)

        # Get color from COLOR_CONFIGS, default to a color if not found
        color = COLOR_CONFIGS.get(approach, f'C{i}')

        # Plot original data
        ax.plot(x_vals, y_vals, marker=markers[i], linestyle='--', color=color, label=approach)

        # Fit power law
        try:
            params, _ = curve_fit(power_law, x_vals, y_vals, maxfev=10000)
            y_fit = power_law(x_vals, *params)
            r2 = r_squared(y_vals, y_fit)
            fit_results.append((approach, params[0], params[1], params[2], r2))

            # Plot fitted curve
            x_dense = np.linspace(min(x_vals), max(x_vals), 200)
            y_dense = power_law(x_dense, *params)
            ax.plot(x_dense, y_dense, '-', color=color, alpha=0.6)
        except RuntimeError:
            print(f"Fit failed for {approach}")
            continue

    #ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    #ax.set_xticklabels([f'$2^{int(np.log2(k))}$' for k in k_values])
    ax.set_xlabel("Number of Generations (k)", fontsize=12)
    ax.set_ylabel("Selection Accuracy (%)", fontsize=12)
    ax.set_title(f"Power Law Fits - {dataset_name} ({model_size})", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"powerlaw_fit_{dataset_name}_{model_size}.png")
    plt.close()

    # Print fitted parameters
    print(f"Fitted power law parameters for {dataset_name} ({model_size}):")
    print("Approach".ljust(40) + "a".rjust(10) + "b".rjust(10) + "c".rjust(10) + "R^2".rjust(10))
    print("-" * 80)
    for name, a, b, c_, r2 in fit_results:
        print(f"{name.ljust(40)}{a:10.3f}{b:10.3f}{c_:10.3f}{r2:10.4f}")
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with dummy data')
    parser.add_argument('--dataset', type=str, action='append', help='Dataset name or comma-separated list of datasets')
    parser.add_argument('--model_size', type=str, action='append', help='Model size (8B, 70B, or both)')
    parser.add_argument('--verifier_size', type=str, default='all', help='Verifier size')
    parser.add_argument('--grid', action='store_true', help='Create a grid plot of all datasets')
    parser.add_argument('--include_tensor_decomp', action='store_true', help='Include tensor decomposition variant')
    parser.add_argument('--include_no_drop', action='store_true', help='Include no verifier dropping variant')
    parser.add_argument('--include_naive_ensemble', action='store_true', help='Include naive ensemble variant')
    parser.add_argument('--add_power_law', action='store_true', default=True, help='Add power law fit to plot')
    parser.add_argument('--plot_scale_verifier', action='store_true',  default=False, help='Plot accuracy vs number of verifiers')
    parser.add_argument('--plot_accuracy_heatmap', action='store_true', default=False, help='Plot accuracy heatmap')

    args = parser.parse_args()

    if args.debug:
        if args.dataset and args.model_size:
            debug_plot(args.dataset[0], args.model_size[0], args.verifier_size)
        else:
            # Run debug mode for all datasets and model sizes
            for dataset_name in ALL_DATASETS:
                for model_size in ALL_MODEL_SIZES:
                    debug_plot(dataset_name, model_size, 'all')
    else:
        # Parse datasets and model sizes
        combinations = []
        if args.dataset and args.model_size:
            for dataset_str, model_size_str in zip(args.dataset, args.model_size):
                # Parse datasets
                datasets = [d.strip() for d in dataset_str.split(',')]
                # Validate datasets
                invalid_datasets = [d for d in datasets if d not in ALL_DATASETS]
                if invalid_datasets:
                    print(f"Warning: Invalid datasets specified: {invalid_datasets}")
                    print(f"Valid datasets are: {ALL_DATASETS}")
                    datasets = [d for d in datasets if d in ALL_DATASETS]

                # Parse model sizes
                if model_size_str.lower() == 'both':
                    model_sizes = ['8B', '70B']
                else:
                    model_sizes = [model_size_str]
                    if model_size_str not in ALL_MODEL_SIZES:
                        print(f"Warning: Invalid model size specified: {model_size_str}")
                        print(f"Valid model sizes are: {ALL_MODEL_SIZES}")
                        model_sizes = ALL_MODEL_SIZES

                # Add all combinations
                for dataset in datasets:
                    for model_size in model_sizes:
                        combinations.append((dataset, model_size))
        else:
            # Default to all combinations
            for dataset_name in ALL_DATASETS:
                for model_size in ALL_MODEL_SIZES:
                    combinations.append((dataset_name, model_size))

        verifier_sizes = ['all']    
        for dataset_name, model_size in combinations:
            for verifier_size in verifier_sizes:
                if args.grid:
                    # If grid mode is enabled, plot all datasets for this model size in a grid
                    plot_grid(ALL_DATASETS, model_size, verifier_size, include_tensor_decomp=args.include_tensor_decomp, include_no_drop=args.include_no_drop, include_naive_ensemble=args.include_naive_ensemble)
                    break  # Only need to do this once per model size
                else:
                    main(dataset_name, model_size, verifier_size, include_tensor_decomp=args.include_tensor_decomp, include_no_drop=args.include_no_drop, include_naive_ensemble=args.include_naive_ensemble)
                
                if args.plot_scale_verifier:
                    plot_verifier_scaling(dataset_name, model_size, verifier_size, order="from_best")
                    plot_verifier_scaling(dataset_name, model_size, verifier_size, order="from_worst")

                if args.plot_accuracy_heatmap:
                    plot_accuracy_heatmap_verifiers_vs_samples(dataset_name, model_size, verifier_size, order="from_best")
                    plot_accuracy_heatmap_verifiers_vs_samples(dataset_name, model_size, verifier_size, order="from_worst")
                    plot_accuracy_heatmap_verifiers_vs_samples(dataset_name, model_size, verifier_size, order="random")
