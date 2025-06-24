"""
Plot coverage of a model in the oracle setting.
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


def main(dataset_name, model_size, verifier_size, num_bootstrap_samples=1, seed=0):
    rng = np.random.default_rng(seed)

    data_cfg = get_data_cfg(verifier_size)
    dataset = VerificationDataset(dataset_name, model_size, **data_cfg)
    (x_train, y_train) = dataset.test_data
    train_answers = dataset.test_answers

    # kvalues log 2 between 1 and y_train
    num_responses = y_train.shape[1]
    k_values = [2**i for i in range(1, int(np.log2(num_responses))+1)]
    k_values.append(num_responses)

    majority_topk_values = [1] #, 2, 5, 10]

    # Bootstrap
    B = num_bootstrap_samples
    results_majority = {m: np.zeros((B, len(k_values))) for m in majority_topk_values}
    results_lr = np.zeros((B, len(k_values)))
    results_lr_per_problem = np.zeros((B, len(k_values)))
    results_best_verifier = np.zeros((B, len(k_values)))
    results_pass_at_k = np.zeros((B, len(k_values)))

    num_problems, num_responses, num_verifiers = x_train.shape

    for k_idx, k in enumerate(k_values):
        for b in range(B):
            x_sampled = np.zeros((num_problems, k, num_verifiers))
            y_sampled = np.zeros((num_problems, k))
            answers_sampled = []

            for i in range(num_problems):
                idx = rng.choice(num_responses, size=k, replace=False)
                x_sampled[i] = x_train[i, idx]
                y_sampled[i] = y_train[i, idx]
                answers_sampled.append([train_answers[i][j] for j in idx])

            # Majority@k
            for m in majority_topk_values:
                acc, _ = calculate_majority_M_at_k(
                    answers_sampled, y_sampled, k, topM=m, return_mean=True
                )
                results_majority[m][b, k_idx] = acc

            # OracleLR (per-data)
            results_lr[b, k_idx] = vanilla_lr([k], x_sampled, y_sampled)[0]


            # OracleLR (per-problem)
            results_lr_per_problem[b, k_idx] = vanilla_lr_per_problem([k], x_sampled, y_sampled)[0]

            # Best verifier
            results_best_verifier[b, k_idx] = get_best_verifier(x_sampled, y_sampled)
    
            # Pass@K unbiased
            results_pass_at_k[b, k_idx] = calculate_pass_k_gt(y_sampled, [k])[k]
    
    # Pass@K biased
    pass_at_k_results = np.mean(results_pass_at_k, axis=0)
    # Let's fix this a bit 
    pass_at_k_results_unbiased = calculate_pass_k_unbiased(y_train, k_values, return_mean=True)
    pass_at_k_results_unbiased = [pass_at_k_results_unbiased[k] for k in k_values]

    # --- Plot results

    plt.plot(k_values, pass_at_k_results, 'o--',label="Pass@k", color='gray')

    # Add majority@k
    for m in majority_topk_values:
        mean = np.mean(results_majority[m], axis=0)
        err = np.std(results_majority[m], axis=0)
        plt.plot(k_values, mean, 'o--', label=f"Majority{m}@k")
        plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

    # Add OracleLR (per data)
    mean = np.mean(results_lr, axis=0)
    err = np.std(results_lr, axis=0)
    plt.plot(k_values, mean, 'o--', label="OracleLR (per data)")
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

    # Add OracleLR (per problem)
    mean = np.mean(results_lr_per_problem, axis=0)
    err = np.std(results_lr_per_problem, axis=0)
    plt.plot(k_values, mean, 'o--', label="OracleLR (per problem)")
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

    # Add Best Verifier
    mean = np.mean(results_best_verifier, axis=0)
    err = np.std(results_best_verifier, axis=0)
    plt.plot(k_values, mean, 'o--', label="Best Verifier")
    plt.fill_between(k_values, mean - err, mean + err, alpha=0.2)

    # Plot coverage
    plt.xlabel('Number of attempts k')
    plt.title(f'{dataset_name} {model_size} Coverage (Verifier Size: {verifier_size})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(FIGURE_PATH / f"coverage_{dataset_name}_{model_size}_verifier{verifier_size}.png")
    plt.close()

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

    return


def vanilla_lr(k_values, x_train, y_train):
    oracle_lr = []
    for k_ in k_values:
        x_train2 = x_train.copy()
        y_train2 = y_train.copy()

        # Select the first k_ attempts:
        # NOTE: This is a biased estimator because it assumes samples are iid
        x_train2 = x_train2[:, :k_, :]
        y_train2 = y_train2[:, :k_]

        num_queries, num_responses, num_verifiers = x_train2.shape

        # Flatten the data
        X = x_train2.reshape((num_queries*num_responses, num_verifiers))
        y = np.array(y_train2).reshape((num_queries*num_responses,))

        # Fit a LR model to all the data 
        model = LogisticRegression(class_weight="balanced")
        model.fit(X, y)

        # Check accuracy:
        # 1. Predict class labels and probabilities
        y_pred = model.predict(X) # (num_queries*num_responses, )
        y_proba = model.predict_proba(X)[:, 1]  # probability for class 1

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

        acc = correct / total if total > 0 else 0.0
        oracle_lr.append(acc)

    return oracle_lr


def get_best_verifier(x_train, y_train):
    """
    """
    # Get the verifier with the highest selection accuracy
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

    # ----- Get verifier scores:
    verifier_f1_scores = np.zeros(num_verifiers)
    for v in range(num_verifiers):
        scores = x_train[:, :, v]
        preds = (scores >= 0.5).astype(int)
        f1 = f1_score_numpy(y_train, preds)
        verifier_f1_scores[v]= f1
    
    return verifier_f1_scores


def verifier_generation_vs_ensemble_size(x_train, y_train, verifier_indices):
    num_queries, num_responses, num_verifiers = x_train.shape

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

def make_plot_law(accuracy_matrix, all_responses, all_verifiers, fig_name, dataset_name, model_size, verifier_size, extra=""):
    # Set up figure
    plt.figure(figsize=(10, 6))

    Z = accuracy_matrix.T
    cmap = cm.get_cmap("plasma")  # You can change to "plasma", "coolwarm", etc.
    norm = mcolors.Normalize(vmin=0, vmax=len(all_verifiers) - 1)

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


if __name__ == "__main__":
    
    verifier_sizes = ['8', '80']    
    for dataset_name in ALL_DATASETS:
        for model_size in ALL_MODEL_SIZES:
            for verifier_size in verifier_sizes:
                main(dataset_name, model_size, verifier_size)
