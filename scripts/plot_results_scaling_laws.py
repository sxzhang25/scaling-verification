"""
Script to plot scaling laws for verifier models.
Sweeps:

Majority Vote and Coverage:
- fjgopp24: includes pass@k and majority vote w seed 0, model_size 8B,70B verifier_size 8
- gndsmj1x: includes pass@k and majority vote w seeds 1,2, model_size 8B,70B verifier_size 8 
- : include verifier_size=80

LR model
- ba9wa2ad: LR per problem and per dataset
- 56cpy1hc: LR with <80b verifiers

Bert Embeddings:
- vpwuinh7: includes bert model stacked.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_PATH = PROJECT_ROOT / "figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


def process_sweep(api, entity, project, sweep_id):
    """Fetch and process W&B sweep runs into a list of records."""
    if sweep_id in ["fjgopp24", "gndsmj1x"]:
        entity = "ekellbuch"  # override entity if needed

    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    records, skipped = [], 0

    for run in sweep.runs:
        try:
            config = run.config
            summary = run.summary
            records.append({
                "dataset": config["data_cfg.dataset_name"],
                "model_size": config["data_cfg.model_size"],
                "train_split": config["data_cfg.train_split"],
                "train_samples": config["data_cfg.train_samples"],
                "seed": config["data_cfg.random_seed"],
                "model_class": config.get("model_cfg.model_class", ""),
                "verifier_size": config["verifier_cfg.verifier_size"],
                "model_type": config["model_cfg.model_type"],
                "test_select_accuracy": summary.get("epoch_test_select_accuracy"),
                "train_select_accuracy": summary.get("epoch_train_select_accuracy"),
                "model_lr": config.get("fit_cfg.lr", ""),
            })
        except KeyError as e:
            logger.warning(f"Skipping run {run.id}: missing key {e}")
            skipped += 1

    logger.info(f"Processed {len(records)} runs from sweep {sweep_id}, skipped {skipped}")
    return records


def plot_subset(ax, group, x, y, yerr, label, color):
    ax.errorbar(
        group[x], group[y], yerr=group[yerr],
        label=label, capsize=3, marker='o', color=color
    )


def plot_panel(ax, split_data, y_col, yerr_col, title):
    """Plot a panel for a specific train_split and y-axis type."""
    for model_type, group in split_data.groupby("model_type"):
        group = group.sort_values("train_samples")

        if model_type == "coverage":
            plot_subset(ax, group, "train_samples", y_col, yerr_col, "coverage", "grey")
        elif model_type == "majority_vote":
            plot_subset(ax, group, "train_samples", y_col, yerr_col, "majority_vote", "blue")
        elif model_type == "logistic_regression":
            for model_class, subgroup in group.groupby("model_class"):
                color = "#ff7f0e" if model_class == "per_dataset" else "green"
                label = f"logistic_regression - {model_class}"
                plot_subset(ax, subgroup.sort_values("train_samples"), "train_samples", y_col, yerr_col, label, color)
        elif model_type == "bert_classifier":
            for model_lr, subgroup in group.groupby("model_lr"):
                color = "purple" if model_lr == 0.0005 else "plum"
                label = f"bert - lr = {model_lr}"
                plot_subset(ax, subgroup, "train_samples", y_col, yerr_col, label, color)
        else:
            plot_subset(ax, group, "train_samples", y_col, yerr_col, model_type, None)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Train Samples")
    ax.set_ylabel("Selection Accuracy")
    ax.set_title(title)
    ax.legend()


def main(entity, project, sweep_ids):
    api = wandb.Api()

    # Collect all records from all sweeps
    all_records = []
    for sweep_id in sweep_ids:
        all_records.extend(process_sweep(api, entity, project, sweep_id))

    my_sweep_ids = '_'.join(sweep_ids)
    if not all_records:
        logger.error("No valid data found.")
        return

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.error("No valid data found after DataFrame construction.")
        return

    # First aggregate stats for each configuration including learning rate
    grouped = df.groupby([
        "dataset", "model_size", "train_split", "model_class",
        "train_samples", "model_type", "verifier_size", "model_lr"
    ]).agg(
        mean_test_select_predict=("test_select_accuracy", "mean"),
        std_test_select_predict=("test_select_accuracy", "std"),
        mean_train_select_predict=("train_select_accuracy", "mean"),
        std_train_select_predict=("train_select_accuracy", "std")
    ).reset_index()

    sns.set(style="whitegrid")

    for dataset in grouped["dataset"].unique():
        for model_size in grouped["model_size"].unique():
            for verifier_size in grouped["verifier_size"].unique():
                subset = grouped[
                    (grouped["dataset"] == dataset) &
                    (grouped["model_size"] == model_size) &
                    (grouped["verifier_size"] == verifier_size)
                ]
                if subset.empty:
                    logger.warning(f"No data for {dataset} {model_size} {verifier_size}")
                    continue

                fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey="row")
                fig.suptitle(f"{dataset} - Model Size: {model_size} - Verifier Size: {verifier_size}")

                split_1 = subset[subset["train_split"] == 1.0]
                split_08 = subset[subset["train_split"] == 0.8]

                plot_panel(axes[0, 0], split_1, "mean_train_select_predict", "std_train_select_predict", "Train Select Accuracy (Split = 1.0)")
                plot_panel(axes[0, 1], split_08, "mean_train_select_predict", "std_train_select_predict", "Train Select Accuracy (Split = 0.8)")
                plot_panel(axes[1, 0], split_1, "mean_test_select_predict", "std_test_select_predict", "Test Select Accuracy (Split = 1.0)")
                plot_panel(axes[1, 1], split_08, "mean_test_select_predict", "std_test_select_predict", "Test Select Accuracy (Split = 0.8)")

                plt.tight_layout()
                plt.savefig(FIGURE_PATH / f"scaling_laws_sweep_{my_sweep_ids}_{dataset}_{model_size}_verifier_{verifier_size}.png")
                plt.close()
    print(f"FInished {my_sweep_ids}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot verifier scaling laws from W&B sweeps")
    parser.add_argument("--entity", default="hazy-research", help="W&B entity")
    parser.add_argument("--project", default="verification", help="W&B project")
    # LR models, MV models, BERT models
    parser.add_argument("--sweeps", nargs="+", default=["fjgopp24", "gndsmj1x", "ba9wa2ad", "56cpy1hc", "vpwuinh7"], help="Sweep IDs")

    args = parser.parse_args()
    main(args.entity, args.project, args.sweeps)
