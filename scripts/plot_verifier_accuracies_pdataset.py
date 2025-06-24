"""
Plot characteristics of the data
"""
from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(PROJECT_ROOT)
FIGURE_PATH = PROJECT_ROOT / "figures"
os.makedirs(FIGURE_PATH, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
try:
    import weaver
except:
    sys.path.append(str(PROJECT_ROOT))
from weaver.dataset import VerificationDataset
from weaver.constants import DATASET_TO_HF
from weaver.utils_metrics import calculate_pass_k_unbiased, calculate_pass_k_gt, calculate_majority_M_at_k
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


DATA_IGNORE_LIST = ["CodeContests_gonly"]
ALL_DATASETS = list(set(DATASET_TO_HF.keys()) - set(DATA_IGNORE_LIST))
ALL_DATASETS = [d for d in ALL_DATASETS if "v2" in d]

ALL_MODEL_SIZES = list(set(DATASET_TO_HF[ALL_DATASETS[0]].keys()) - set(DATA_IGNORE_LIST))

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
    data_cfg = {
        "train_split": 1.0,
        "train_queries": 1,
        "train_samples": 1,
        "random_seed": 0,
        "nan_replacement": 0,
        "reward_threshold": None,
        "normalize_type": "per_problem",
        "normalize_method": "minmax",
        "closest_train_problem_method": "mean_verifier_distance",
        "closest_train_problem_metric_type": "euclidean",
        "verifier_cfg": get_verifier_cfg("all", verifier_size, []),
        "mv_as_verifier": True,
        "normalize_params": {
            "output_distribution": "normal",
            "n_quantiles": 100,
        }
    }
    return data_cfg


def plot_all_datasets(model_size, verifier_size, plot_type="selection_accuracy"):
    n_datasets = len(ALL_DATASETS)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, dataset_name in enumerate(ALL_DATASETS):
        ax = axes[idx]
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)

        X_data, y_data = dataset.test_data
        # X_data is verifier score: num_problems, num_samples, num_verifiers

        # Selection accuracy per verifier
        # Calculate the sample with the highest score per problem
        if plot_type == "selection_accuracy":
            top1_sample_idx = np.argmax(X_data, 1) # highest scoring sample per problem, (num_problems, num_verifiers)
            
            P, V = top1_sample_idx.shape
            rows = np.arange(P)[:, None]         # shape (P, 1)
            cols = top1_sample_idx               # shape (P, V)
            y_top1 = y_data[rows, cols]          # shape (P, V)

            # Check if that sample is positive
            acc_matrix = (y_top1 == 1) # (num_problems, num_verifiers)

        # Calculate the average accuracy of the problem for each verifier
            vmin, vmax = 0, 1
        elif plot_type == "average_accuracy":
            y_pred = (X_data >= 0.5).astype(int)
            acc_matrix = (y_pred == y_data[..., None]).mean(axis=1)  # (problems × verifiers)
            vmin, vmax = 0, 1
        elif plot_type == "variance_of_accuracy":
            y_pred = (X_data >= 0.5).astype(int) # (num_problems, num_samples, num_verifiers)
            acc_matrix = (y_pred == y_data[..., None]).var(axis=1)  # (problems × verifiers)
            vmin, vmax = acc_matrix.min(), acc_matrix.max()
        elif plot_type == "score_variance":
            acc_matrix = X_data.var(axis=1)
            vmin, vmax = acc_matrix.min(), acc_matrix.max()
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")
        
        #acc_matrix = np.nan_to_num(acc_matrix, nan=np.nanmean(acc_matrix, axis=0))

        # Apply clustering to reorder rows and columns
        row_order = leaves_list(linkage(acc_matrix, method='ward'))
        col_order = leaves_list(linkage(acc_matrix.T, method='ward'))

        acc_matrix_clustered = acc_matrix[row_order][:, col_order]
        verifier_labels = np.array(dataset.verifier_names)[col_order]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        sns.heatmap(
            acc_matrix_clustered,
            ax=ax,
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
            xticklabels=verifier_labels,
            yticklabels=False,
            cbar=True,
            cbar_ax=cax
        )
        num_ticks= 5
        tick_vals = np.linspace(vmin, vmax, num=num_ticks)
        tick_labels = [f"{val:.1f}" for val in tick_vals]

        # Set min/max ticks on colorbar
        cax.set_yticks(tick_vals)   
        cax.set_yticklabels(tick_labels)

        ax.set_title(dataset_name)
        ax.set_xlabel("Verifier")
        ax.set_ylabel("Problems")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)

    # Remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Verifier {plot_type} Clustering (Model: {model_size}, Verifier size: {verifier_size})",
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"clustered_verifier_{plot_type}_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting all datasets {model_size} {verifier_size} {plot_type}\n")
    return


def calculate_inverse_covariance_matrix(model_size, verifier_size):
    """
    Calculate and return the inverse covariance matrix of verifier scores
    (aggregated across problems and samples).
    """
    results = {}
    for dataset_name in ALL_DATASETS:
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        X_data, _ = dataset.test_data  # shape: (num_problems, num_samples, num_verifiers)

        # Reshape into [num_problems * num_samples, num_verifiers]
        X_flat = X_data.reshape(-1, X_data.shape[-1])  # shape: [P*S, V]

        # Optional: Normalize to zero mean and unit variance per verifier
        X_centered = X_flat - np.mean(X_flat, axis=0, keepdims=True)

        # Covariance matrix
        cov = np.cov(X_centered, rowvar=False)  # shape: [V, V]

        # Inverse covariance (precision matrix)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print(f"[Warning] Covariance matrix for {dataset_name} is singular, using pseudo-inverse.")
            inv_cov = np.linalg.pinv(cov)

        results[dataset_name] = {
            "cov": cov,
            "inv_cov": inv_cov,
            "verifier_names": dataset.verifier_names,
        }

    return results


def plot_inverse_covariance_matrices(model_size, verifier_size):
    inv_covs = calculate_inverse_covariance_matrix(model_size, verifier_size)

    n_datasets = len(inv_covs)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, (dataset_name, data) in enumerate(inv_covs.items()):
        ax = axes[idx]
        inv_cov = data["inv_cov"]
        verifier_names = data["verifier_names"]

        # Apply hierarchical clustering
        row_order = leaves_list(linkage(inv_cov, method='ward'))
        inv_cov_clustered = inv_cov[row_order][:, row_order]
        verifier_names_clustered = np.array(verifier_names)[row_order]

        off_diag = inv_cov[~np.eye(inv_cov.shape[0], dtype=bool)]
        vmin = np.percentile(off_diag, 1)
        vmax = np.percentile(off_diag, 99)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        sns.heatmap(
            inv_cov_clustered,
            ax=ax,
            cmap="RdBu",
            center=0.0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=verifier_names_clustered,
            yticklabels=verifier_names_clustered,
            square=True,
            cbar=True,
            cbar_ax=cax
        )


        num_ticks = 5
        tick_vals = np.linspace(vmin, vmax, num=num_ticks)
        tick_labels = [f"{val:.1f}" for val in tick_vals]

        # Set min/max ticks on colorbar
        cax.set_yticks(tick_vals)   
        cax.set_yticklabels(tick_labels)

        ax.set_title(f"{dataset_name} (Inverse Covariance)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Verifier Inverse Covariance Matrix (Model: {model_size}, Verifier size: {verifier_size})",
                 fontsize=16, y=1.02)


    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"inverse_covariance_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting inverse covariance matrices {model_size} {verifier_size}\n")
    return

def plot_partial_correlation_matrices(model_size, verifier_size):
    inv_covs = calculate_inverse_covariance_matrix(model_size, verifier_size)

    n_datasets = len(inv_covs)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, (dataset_name, data) in enumerate(inv_covs.items()):
        ax = axes[idx]
        inv_cov = data["inv_cov"]
        verifier_names = data["verifier_names"]

        # === Compute partial correlation ===
        D = np.sqrt(np.diag(inv_cov))
        partial_corr = -inv_cov / np.outer(D, D)
        np.fill_diagonal(partial_corr, 1.0)

        # === Cluster rows and columns ===
        row_order = leaves_list(linkage(partial_corr, method='ward'))
        partial_corr_clustered = partial_corr[row_order][:, row_order]
        verifier_names_clustered = np.array(verifier_names)[row_order]

        vmin, vmax = -1, 1  # standardized range for partial correlation

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        sns.heatmap(
            partial_corr_clustered,
            ax=ax,
            cmap="RdBu",
            center=0.0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=verifier_names_clustered,
            yticklabels=verifier_names_clustered,
            square=True,
            cbar=True,
            cbar_ax=cax
        )

        tick_vals = np.linspace(vmin, vmax, 5)
        tick_labels = [f"{val:.1f}" for val in tick_vals]
        cax.set_yticks(tick_vals)
        cax.set_yticklabels(tick_labels)

        ax.set_title(f"{dataset_name} (Partial Corr.)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Verifier Partial Correlation Matrix (Model: {model_size}, Verifier size: {verifier_size})",
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"partial_correlation_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting partial correlation matrices {model_size} {verifier_size}\n")
    return


def plot_class_distribution(model_size, verifier_size, plot_style="stacked_bar"):
    """
    Plot class distribution (positive/negative) per problem across all datasets.
    Available plot styles:
      - "stacked_bar"
      - "histogram"
      - "violin"
    """
    from collections import defaultdict
    import pandas as pd

    n_datasets = len(ALL_DATASETS)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    if plot_style in {"stacked_bar", "histogram"}:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, dataset_name in enumerate(ALL_DATASETS):
            ax = axes[idx]
            data_cfg = get_data_cfg(verifier_size)
            dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
            _, y_data = dataset.test_data  # (num_problems, num_samples)

            pos_frac = (y_data == 1).mean(axis=1)
            problem_ids = np.arange(len(pos_frac))

            if plot_style == "stacked_bar":
                ax.bar(problem_ids, pos_frac, label="Positive %", color="tab:blue")
                ax.bar(problem_ids, 1 - pos_frac, bottom=pos_frac, label="Negative %", color="tab:red")
                ax.set_ylim(0, 1)
                if idx == 0:
                    ax.legend()
            elif plot_style == "histogram":
                ax.hist(pos_frac, bins=20, range=(0, 1), color="skyblue", edgecolor="black")
                ax.set_xlabel("Positive Label Fraction")
                ax.set_ylabel("# Problems")

            ax.set_title(dataset_name)
            ax.set_xlabel("Problem ID" if plot_style == "stacked_bar" else "")
            ax.set_ylabel("Count" if plot_style == "stacked_bar" else "")
            ax.tick_params(axis='x', labelsize=6)

        # Remove unused axes
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Class Distribution per Problem ({plot_style}, Model: {model_size}, Verifier size: {verifier_size})",
                     fontsize=16, y=1.02)

        plt.tight_layout()
        plt.savefig(
            FIGURE_PATH / f"class_distribution_{plot_style}_ALL_{model_size}_{verifier_size}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    elif plot_style == "violin":
        # Collect data for seaborn violin plot
        import seaborn as sns
        data = []
        for dataset_name in ALL_DATASETS:
            data_cfg = get_data_cfg(verifier_size)
            dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
            _, y_data = dataset.test_data
            pos_frac = (y_data == 1).mean(axis=1)
            for p in pos_frac:
                data.append({"dataset": dataset_name, "positive_rate": p})

        df = pd.DataFrame(data)

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="dataset", y="positive_rate", inner="quartile", scale="width")
        plt.title(f"Positive Class Rate Distribution by Dataset\n(Model: {model_size}, Verifier size: {verifier_size})")
        plt.ylabel("Positive Label Fraction")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(
            FIGURE_PATH / f"class_distribution_violin_ALL_{model_size}_{verifier_size}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    else:
        raise ValueError(f"Invalid plot_style: {plot_style}")

    print(f"Finished plotting class distribution ({plot_style}) for model {model_size}, verifier {verifier_size}\n")


def plot_class_distribution_summary(model_size, verifier_size):
    """
    Generates:
    1. Histogram of per-problem positive label fraction (one per dataset)
    2. Violin plot comparing positive rate distribution across datasets
    """
    import seaborn as sns
    import pandas as pd

    # ===== Histogram setup =====
    n_datasets = len(ALL_DATASETS)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_hist = axes_hist.flatten()

    # ===== Collect data for violin =====
    all_pos_data = []

    for idx, dataset_name in enumerate(ALL_DATASETS):
        ax = axes_hist[idx]
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        _, y_data = dataset.test_data

        pos_frac = (y_data == 1).mean(axis=1)  # Fraction of positive samples per problem

        # --- Histogram ---
        ax.hist(pos_frac, bins=20, range=(0, 1), color="skyblue", edgecolor="black")
        ax.set_title(dataset_name)
        ax.set_xlabel("Positive Label Fraction")
        ax.set_ylabel("Num Problems")

        # --- Collect for violin ---
        for p in pos_frac:
            all_pos_data.append({
                "dataset": dataset_name,
                "positive_rate": p
            })

    # Remove unused axes in hist subplot
    for j in range(idx + 1, len(axes_hist)):
        fig_hist.delaxes(axes_hist[j])

    fig_hist.suptitle(f"Histogram of Positive Label Fraction per Problem\n(Model: {model_size}, Verifier size: {verifier_size})",
                      fontsize=16, y=1.02)
    fig_hist.tight_layout()
    fig_hist.savefig(
        FIGURE_PATH / f"class_distribution_histogram_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_hist)

    # ===== Violin plot =====
    df = pd.DataFrame(all_pos_data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="dataset", y="positive_rate", inner="quartile", scale="width", cut=0)
    plt.title(f"Positive Label Fraction by Dataset (Violin)\n(Model: {model_size}, Verifier size: {verifier_size})")
    plt.ylabel("Positive Label Fraction")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"class_distribution_violin_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting class distribution histogram + violin for model {model_size}, verifier {verifier_size}\n")
    return


# Add-on 2: Ensemble prediction variance vs. number of verifiers
def plot_ensemble_variance_vs_k_summary(model_size, verifier_size, max_k=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset_name in ALL_DATASETS:
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        X_data, y_data = dataset.test_data

        verifier_scores = X_data.mean(axis=1)  # shape: (P, V)

        # rank from best to worst verifier
        verifier_accuracy = ((X_data >= 0.5).astype(int)  == y_data[..., None]).mean(1).mean(0)
        ranked_idx = np.argsort(verifier_accuracy)[::-1]
        verifier_scores_sorted = verifier_scores[:, ranked_idx]

        P, V = verifier_scores.shape
        if max_k is None:
            max_k = V

        vars = []
        for k in range(1, max_k + 1):
            # average prediction of top k verifiers
            avg_probs = verifier_scores_sorted[:, :k].mean(axis=1)
            # variance of bernoulli distribution: 
            # maximal when p = 0.5 (i.e. when the prediction is most uncertain)
            # minimal when p = 0 or 1 (i.e. when the prediction is most certain)
            variances = avg_probs * (1 - avg_probs)
            vars.append(np.mean(variances))

        min_idx = np.argmin(vars) + 1
        optimal_k = vars[min_idx - 1]

        ax.plot(range(1, max_k + 1), vars, label=dataset_name)
        ax.plot(min_idx, optimal_k, 'ro')
        ax.set_xticks(range(1, max_k + 1, 2))

    ax.set_title(f"Ensemble Prediction Variance vs. Number of Verifiers\n(Model: {model_size}, Verifier size: {verifier_size})")
    ax.set_xlabel("Number of Verifiers in Ensemble (from best to worst)")
    ax.set_ylabel("Average Bernoulli Variance")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"ensemble_variance_vs_k_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting ensemble variance vs verifier count for model {model_size}, verifier {verifier_size}\n")
    return


# Add-on 3: Agreement matrix heatmap (e.g. for GPQA only)
def plot_verifier_agreement_matrix(model_size, verifier_size):
    n_datasets = len(ALL_DATASETS)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, dataset_name in enumerate(ALL_DATASETS):
        ax = axes[idx]

        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        X_data, _ = dataset.test_data
        verifier_scores = X_data.mean(axis=1) > 0.5  # binarized predictions
        V = verifier_scores.shape[1]

        agreement = np.zeros((V, V))
        for i in range(V):
            for j in range(V):
                agreement[i, j] = np.mean(verifier_scores[:, i] == verifier_scores[:, j])

        row_order = leaves_list(linkage(agreement, method='ward'))
        agreement_clustered = agreement[row_order][:, row_order]
        verifier_names = np.array(dataset.verifier_names)[row_order]

        sns.heatmap(agreement_clustered,
                    xticklabels=verifier_names,
                    yticklabels=verifier_names,
                    cmap="Blues", vmin=0, vmax=1, square=True,
                    ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        ax.tick_params(axis='both', labelsize=6, pad=1)
        ax.set_title(f"{dataset_name}")

    # suptitle
    plt.suptitle(f"Verifier Agreement Matrix (Model: {model_size}, Verifier size: {verifier_size})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(
    FIGURE_PATH / f"verifier_agreement_matrix_{model_size}_{verifier_size}.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting verifier agreement matrix for model {model_size}, verifier {verifier_size}")

# Add-on 4: Ensemble margin distribution
def plot_ensemble_margin_distribution(model_size, verifier_size):
    n_datasets = len(ALL_DATASETS)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, dataset_name in enumerate(ALL_DATASETS):
        ax = axes[idx]
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        X_data, _ = dataset.test_data

        verifier_votes = X_data.mean(axis=1) > 0.5  # (P, V), binarized
        V = verifier_votes.shape[1]
        vote_sums = verifier_votes.sum(axis=1)
        margins = np.abs(vote_sums - (V - vote_sums))

        ax.hist(margins, bins=20, color="darkorange", edgecolor="black")
        ax.set_title(dataset_name)
        ax.set_xlabel("Vote Margin")
        ax.set_ylabel("Number of Problems")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Ensemble Vote Margin Distribution (Model: {model_size}, Verifier size: {verifier_size})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"ensemble_margin_distribution_ALL_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting ensemble margin distributions for model {model_size}, verifier {verifier_size}")


# Plot lower bound: 
def plot_lr_lower_bound(model_size, verifier_size):

    num_datasets = len(ALL_DATASETS)
    n_rows = num_datasets
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for d_idx, dataset_name in enumerate(ALL_DATASETS):
        data_cfg = get_data_cfg(verifier_size)
        dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
        X_data, y_data = dataset.test_data


        num_queries, num_responses, num_verifiers = X_data.shape

        # Flatten the data
        X = X_data.reshape((num_queries*num_responses, num_verifiers))
        y = np.array(y_data).reshape((num_queries*num_responses,))

        # Fit a LR model to all the data 
        model = LogisticRegression(class_weight="balanced")
        model.fit(X, y)

        # Check accuracy:
        # 1. Predict class labels and probabilities:
        y_pred_flat = model.predict(X) # (num_queries*num_responses, )
        y_probs_flat = model.predict_proba(X) #(num_queries*num_responses, 2)

        y_probs1_flat = model.predict_proba(X)[:, 1]  # probability for class 1 #(num_queries*num_responses, )
        #print('Accuracy:', accuracy_score(y, y_pred))
        # Select sample with the highest score for each answer
        y_pred = y_pred_flat.reshape((num_queries, num_responses))
        y_probs1 = y_probs1_flat.reshape((num_queries, num_responses))

        eps = 1e-6
        y_probs1 = np.clip(y_probs1, eps, 1 - eps)

        # Select index of top candidate by predicted probability
        top_index = np.argmax(y_probs1, 1) #(num_queries, )

        # is the top-ranked candidate according to the model correct?   
        pred_labels = y_pred[np.arange(len(top_index)), top_index] # pred_labels

        # Negative log likelihood
        nll = - (y_data* np.log(y_probs1) + (1 - y_data) * np.log(1 - y_probs1))
        # sklearn LR adds L2 regulation by default, so it minimized NLL + \lamba/2 |beta||^2
        # but ok to use this metric?
        # nll2 = log_loss(y, y_probs_flat, normalize=True)
        # p, y, nll, interpretation
        # 0.99, 1, 0.01: excellent (confident and correct)
        # 0.6, 1, 0.51 okay
        # 0.5, 1 or 0, ~0.69 uncertain (max entropy)
        # 0.4, 1, ~0.92 wrong-leaning, bad
        # 0.01, 1, ~4.60 worst case 
        # The models predicts 0.5 for every sample: NLL: n*log(2)= 0.693 n
        
        # Let's get the mean estimate of responses
        bar_yij =  y_data.mean(1) # num_queries

        # stability :
        bar_yij = np.clip(bar_yij, eps, 1- eps)
        # NLL using entropu estimate
        # nll_mean = - (1/R) \sum_j log P(y_ij| \bar{y_ij}) = \bar{y_ij} \log \bar{y_ij} + (1-  \bar{y_ij}) log (1 - \bar{y_ij})
        nll_mean = - (bar_yij * np.log(bar_yij) + (1- bar_yij)* np.log(1-bar_yij))
        # 0: perfect prior knowledge
        
        relative_gain = (nll - nll_mean[..., None])
        # <<0 huge improvement over baseline
        # 0.1 to 0.5 solid gains
        # = 0 baseline match
        # > 0.5 model is worse

        #relative_gain = gain / nll_mean[..., None]
        # 0: model performs exactly as the baseline, no gain
        # < 0: model performs better than the baseline - NLL is lower
        # > 0: model performs worse than the baseline - NLL is higher
        # ~ -1 : model has dramatically better fit, \
        # >1 : model is much worse: NLL is over double baseline.
        # cap execive gains
        #relative_gain = np.clip(relative_gain, -5, 5)

        # rule of thumb:
        # < 0.3 very low NLL: model is confident and correct
        # 0.3-0.7: moderate NLL: model is uncertain
        # 0.7-1.5: model is wrong
        average_gain = relative_gain.mean()

        # CLuster based on gain
        row_order = leaves_list(linkage(relative_gain, method='ward'))
        col_order = leaves_list(linkage(relative_gain.T, method='ward'))

        gain_clustered = relative_gain[row_order][:, col_order]
        nll_clustered = nll[row_order][:, col_order]

        ax = axes[d_idx,0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        sns.heatmap(
            nll_clustered,
            ax=ax,
            center=0.69,
            cmap="RdBu",
            square=False,
            cbar=True,
            cbar_ax=cax,
            yticklabels=False,
            xticklabels=False,
            vmin=-5,
            vmax=5,

        )
        ax.set_title(f"{dataset_name} LR (NLL)")
        ax.set_ylabel("Problems")
        ax.set_xlabel("Samples")

        ax = axes[d_idx,1]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        sns.heatmap(
            gain_clustered,
            ax=ax,
            cmap="RdBu",
            center=0.0,
            square=False,
            cbar=True,
            cbar_ax=cax,
            yticklabels=False,
            xticklabels=False,
        )
        ax.set_title(f"{dataset_name} Gain \n NLL model - H(y_mean) = {np.round(average_gain,3)}")
        ax.set_ylabel("Problems")
        ax.set_xlabel("Samples")

        nll_pred = nll[np.arange(len(top_index)), top_index]
        gain_pred = gain_clustered[np.arange(len(top_index)), top_index]
        y_true = np.any(y_data, axis=1)

        # Optional: Save data for analysis across datasets if looping
        if d_idx == 0:
            all_nll_preds = [nll_pred]
            all_gain_preds = [gain_pred]
            all_names = [dataset_name]
            all_ys = [y_true]
        else:
            all_nll_preds.append(nll_pred)
            all_gain_preds.append(gain_pred)
            all_names.append(dataset_name)
            all_ys.append(y_true)


    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"lr_nll_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Finished plotting lower bound for lr model {model_size}, verifier {verifier_size}\n")

    # Plot NLL and gain for top prediction only
    fig, axes = plt.subplots(len(all_nll_preds), 2, figsize=(10, 2 * len(all_nll_preds)))
    if len(all_nll_preds) == 1:
        axes = np.expand_dims(axes, axis=0)  # make sure axes is 2D

    for i, (nlls, gains, name) in enumerate(zip(all_nll_preds, all_gain_preds, all_names)):
        ax = axes[i, 0]
        ax.hist(nlls, bins=30, color="royalblue", alpha=0.8)
        ax.set_title(f"{name} - NLL (Top Pred)")
        ax.set_xlabel("NLL")
        ax.set_ylabel("Count")

        ax = axes[i, 1]
        ax.hist(gains, bins=30, color="crimson", alpha=0.8)
        ax.set_title(f"{name} - Gain (Top Pred)")
        ax.set_xlabel("Gain (NLL - H(y_mean))")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / f"lr_top_pred_hist_{model_size}_{verifier_size}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return


if __name__ == "__main__":
    verifier_sizes = ["all"]
    for model_size in ALL_MODEL_SIZES:
        for verifier_size in verifier_sizes:
            plot_lr_lower_bound(model_size, verifier_size)