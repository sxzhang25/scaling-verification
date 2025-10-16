import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from weaver.dataset import VerificationDataset, ClusteringDataset, create_df_from_h5
from weaver.models import Model
import wandb
os.environ["WANDB_SILENT"] = "true"
from sklearn.cluster import KMeans
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# TODO: implement this
def save_weaver_model(model, args):
    """Save Weaver model to dataset."""
    if model.model_type != "weak_supervision":
        return

    # Create pickle with all model params.
    model_params = {}

    # def predict_proba(self, X):
    #     X = X[:, self.verifier_idxs] # use only the verifiers that were used to fit the model
    #     if self.use_continuous:
    #         probs = np.zeros((X.shape[0], 2))
    #         for i in range(X.shape[0]):
    #             prob = np.expand_dims(self.Sigma_hat[self.m, :self.m], axis=0) \
    #                                 .dot(np.linalg.inv(self.Sigma_hat[:self.m, :self.m])) \
    #                                 .dot(np.expand_dims(X[i, :self.m], axis=1))
    #             probs[i, 0] = 1 - prob.item()
    #             probs[i, 1] = prob.item()
    #         return probs
    #     else:
    #         return super().predict_proba(X)
    

    output_path =f'{os.path.splitext(args.data_cfg.dataset_path)[0]}_model.pkl'
    with open(output_path, "wb") as f:
        pickle.dump(model_params, f)

def save_weaver_scores_to_dataset(data, model, all_test_results, args):
    """Save Weaver scores/probabilities to dataset for distillation."""
        
    if model.model_type != "weak_supervision":
        return
    
    dataset_path = data.dataset_mapping
    
    print("Saving Weaver scores to dataset...")
    
    if os.path.exists(dataset_path):
        if dataset_path.endswith(".parquet"):
            df = pd.DataFrame(load_dataset("parquet", data_files=dataset_path)['train'])
        elif dataset_path.endswith(".h5"):
            df = create_df_from_h5(dataset_path, args.verifier_cfg.verifier_subset)
        else:
            df = pd.DataFrame(load_from_disk(dataset_path))
    else:
        df = pd.DataFrame(load_dataset(dataset_path)["data"])

    # Compute Weaver class 1 probabilities for each problem
    weaver_scores_list = []
    for _, row in all_test_results.iterrows():
        if 'weaver_sample_scores' in row and row['weaver_sample_scores'] is not None:
            weaver_scores_list.append(row['weaver_sample_scores'])
    
    # Extract Weaver scores from test results
    weaver_scores_list = []
    for _, row in all_test_results.iterrows():
        if 'weaver_sample_scores' in row and row['weaver_sample_scores'] is not None:
            weaver_scores_list.append(row['weaver_sample_scores'])
        else:
            # Fallback: create zeros if scores missing
            # Assuming consistent number of samples per problem
            num_samples = len(df.iloc[0]['samples']) if len(df) > 0 else 10
            weaver_scores_list.append([0.0] * num_samples)
            
    # Map scores back to full dataset using test indices
    full_scores = [None] * len(df)
    for i, scores in enumerate(weaver_scores_list):
        test_idx = data.test_idx[i]  # Original dataset index for this test problem
        full_scores[test_idx] = scores
    
    # Fill missing entries with zeros
    for i in range(len(df)):
        if full_scores[i] is None:
            num_samples = len(df.iloc[i]['samples'])
            full_scores[i] = [0.0] * num_samples
    
    # Save to dataset
    df['weaver_scores'] = full_scores
    modified_dataset = Dataset.from_pandas(df)
    
    if os.path.exists(dataset_path):
        # If this is a local dataset, add the scores directly to the dataset on disk
        if dataset_path.endswith('.parquet') or dataset_path.endswith('.h5'):
            dataset_path = '/'.join(dataset_path.split('/')[:-1])
            DatasetDict({"data": modified_dataset}).save_to_disk(dataset_path)
        else:
            DatasetDict({"data": modified_dataset}).save_to_disk(dataset_path)
        print(f"Added weavers scores to local dataset {data.dataset_name} with path {dataset_path}")
    else:
        # This is a hub dataset, add the weaver scores there
        DatasetDict({"data": modified_dataset}).push_to_hub(dataset_path)
        print(f"Added weavers scores to Huggingface hub dataset {data.dataset_name} with path {dataset_path}")

def get_test_models_indices(data, model, fit_cfg):
    """
    Get the train model to be used for each test problem.

    fit_cfg.fit_type:
    - wclosest_to_train: use the closest train sample to the test sample.
    - search_weights: search for the best weights across all the train problems.

    If we fitted one model then do not do anything.
    """
    num_train_problems = len(data.train_data[0])
    num_test_problems = len(data.test_data[0])

    if num_train_problems == 0:
        if model.model_class in ["per_dataset_cluster"]:
            all_closest_train_idxs = model.clusters.find_test_set_clusters(data)
            return all_closest_train_idxs
        else:
            return [] * num_test_problems

    # If we fitted one model then do not do anything
    if model.model_class not in ["per_problem", "cluster", "per_dataset_cluster"]:
        return [] * num_test_problems

    # If the model is majority vote then do not do anything
    if model.model_type in ["majority_vote"]:
        return [] * num_test_problems

    best_train_indices = np.zeros(num_test_problems) * np.nan

    if model.model_class in ["cluster", "per_dataset_cluster"]:
        # Assign clusters to the test set and reorder the closest train idxs based on the clusters:
        all_closest_train_idxs = model.clusters.find_test_set_clusters(data)
    else:
        all_closest_train_idxs = data.closest_train_idxs

    # Use the closest train sample where the metric used was defined in data
    if fit_cfg.fit_type == "wclosest_to_train":
        for idx in range(num_test_problems):
            ranked_train_idxs = all_closest_train_idxs[idx]
            if model.model_class in ["per_problem", "cluster"]:
                for c_train_idx in ranked_train_idxs:
                    c_group_idx = model.problem_idx_to_group_idx(c_train_idx)
                    if model.is_trained[c_group_idx]:
                        break 
            else:
                assert model.is_trained
                c_train_idx = ranked_train_idxs[0]
            
            best_train_indices[idx] = c_train_idx

    elif fit_cfg.fit_type == "search_weights":
        # For each test set problem, find the best train problem and use its weight.
        num_train_problems = len(data.train_data[0])
        all_trained_models = np.array([model.is_trained[i] for i in range(num_train_problems)])
        trained_models_idxs = np.where(all_trained_models)[0]
        num_trained_models = len(trained_models_idxs)

        # If there are no trained models, then return an empty list
        if num_trained_models == 0:
            return [None] * num_test_problems
        all_train_indices = np.zeros((num_test_problems, num_trained_models)) * np.nan

        X_test, y_test = data.test_data

        for test_idx in range(num_test_problems):
            # for each train model:
            for train_idx in range(num_trained_models):
                problem_idx = trained_models_idxs[train_idx]
                outputs = model.calculate_metrics(X_test[test_idx], y_test[test_idx], problem_idx=problem_idx)
                all_train_indices[test_idx, train_idx] = outputs["top1_positive"]

        best_train_indices = np.argmax(all_train_indices, axis=1)
        best_train_indices = trained_models_idxs[best_train_indices]

    else:
        raise NotImplementedError(f"Unknown fit type: {fit_cfg.fit_type}")

    best_train_indices = best_train_indices.astype(int)
    return best_train_indices


def train_and_evaluate(data, model, fit_cfg):
    """Train and evaluate a model"""
    from rich.console import Console
    console = Console()
    
    all_results, all_test_results = [], []

    X_train, y_train = data.train_data
    X_test, y_test = data.test_data
    x_train_indices = data.train_idx
    x_test_indices = data.test_idx

    train_answers = data.train_answers
    test_answers = data.test_answers

    if model.model_type in ["majority_vote", "first_sample"]:
        X_train = train_answers
        X_test = test_answers

    # ---------------------------------------------------------------------------------------
    # Train model    
    num_train_problems = len(X_train)
    if num_train_problems > 0:
        console.print(f"[yellow]Training model ({model.model_class})...[/yellow]")
        
        if model.model_class in ["per_problem"]:
            for idx in range(num_train_problems):
                X, y = X_train[idx], y_train[idx]
                if model.model_type in ["logistic_regression", "naive_bayes"] and len(np.unique(y)) == 1:
                    continue 
                model.fit(X, y, group_idx=idx)
                
        elif model.model_class in ["per_dataset"]:
            model.fit(X_train, y_train)
            # print("model:", model)
            # print("model \mu:", model.mu)
            # Print all instance class attributes of model
            # console.print("[cyan]Model instance attributes:[/cyan]")
            # for attr in dir(model):
            #     # Exclude methods and built-in attributes
            #     if not attr.startswith("__") and not callable(getattr(model, attr)):
            #         console.print(f"  {attr}: {getattr(model, attr)}")
            
        elif model.model_class == "per_dataset_cluster":
            cluster_idxs = model.clusters.train_cluster_idxs
            model.fit(X_train, y_train, difficulties=cluster_idxs)
            
        elif model.model_class == "cluster":
            for idx in range(len(model.clusters.train_clusters)):
                cluster_idxs = model.clusters.train_clusters[idx]
                X, y = X_train[cluster_idxs], y_train[cluster_idxs]
                if model.model_type in ["logistic_regression", "naive_bayes"] and len(np.unique(y)) == 1:
                    continue
                model.fit(X, y, group_idx=idx)
        else:
            raise NotImplementedError(f"Unknown model class: {model.model_class}")
    else:
        console.print("[yellow]No train data available[/yellow]")

    # ---------------------------------------------------------------------------------------
    # Evaluate on train set
    console.print("[yellow]Evaluating on train set...[/yellow]")
    for idx in range(num_train_problems):
        sample_idx = data.train_idx[idx]
        problem_idx = idx if model.model_class in ["per_problem", "cluster"] else None

        if model.model_class == "per_dataset_cluster":
            cluster_idxs = model.clusters.train_cluster_idxs[idx]
            outputs = model.calculate_metrics(X_train[idx], y_train[idx], difficulties=cluster_idxs)
        else:
            outputs = model.calculate_metrics(X_train[idx], y_train[idx], problem_idx=problem_idx)
            
        if np.isnan(outputs["top1_positive"]):
            continue 
        
        outputs["problem"] = sample_idx
        outputs["set"] = "train"
        outputs["difficulty"] = data.assignments[sample_idx]
        all_results.append(outputs)

    if len(all_results) == 0:
        all_results.append({"sample_accuracy": np.nan, "top1_positive": np.nan, "prediction_accuracy": np.nan,
                           "top1_tp": np.nan, "top1_fp": np.nan, "top1_tn": np.nan, "top1_fn": np.nan, "difficulty": np.nan})
        
    all_results = pd.DataFrame(all_results)

    # ---------------------------------------------------------------------------------------
    # Calculate trained models
    if model.model_class in ["per_problem", "cluster"]:
        num_trained_models = sum(model.is_trained.values())
    elif model.model_class in ["per_dataset", "per_dataset_cluster"]:
        num_trained_models = 1 if model.is_trained else 0
    else:
        raise NotImplementedError(f"Unknown model class: {model.model_class}")

    # ---------------------------------------------------------------------------------------
    # Evaluate on test set
    console.print("[yellow]Evaluating on test set...[/yellow]")
    test_model_indices = get_test_models_indices(data, model, fit_cfg)
    model.is_test = True

    # Fit on test data for weak supervision
    if model.model_type in ["weak_supervision", "unsupervised"]:
        train_not_test = not(float(data.train_split) == 1.0) or not(X_train.shape == X_test.shape) or np.any(X_train != X_test)
        if train_not_test:
            console.print(f"[yellow]Fitting weak supervision model on test set ({len(X_test)} problems)...[/yellow]")
            if model.model_class == "per_dataset":
                model.model.is_test = True
                model.fit(X_test, y_test)
                model.fit_when_calculating_metrics = False
            elif model.model_class == "cluster":
                for idx in range(len(model.clusters.test_clusters)):
                    cluster_idxs = model.clusters.test_clusters[idx]
                    X_test_tmp, y_test_tmp = X_test[cluster_idxs], y_test[cluster_idxs]
                    model.models[idx].is_test = True
                    model.fit(X_test_tmp, y_test_tmp, group_idx=idx)
                    model.models[idx].fit_when_calculating_metrics = False
            elif model.model_class == "per_dataset_cluster":
                cluster_idxs = model.clusters.test_cluster_idxs
                model.model.is_test = True
                model.fit(X_test, y_test, difficulties=cluster_idxs)
                model.model.fit_when_calculating_metrics = False

    # Calculate test metrics
    num_test_problems = len(X_test)
    for idx in range(num_test_problems):
        sample_idx = data.test_idx[idx]
        ranked_train_idxs = data.closest_train_idxs[idx]
        
        if model.model_class in ["per_problem", "cluster"]:
            c_train_idx = test_model_indices[idx]
            if idx != c_train_idx and getattr(data, "same_train_test") and num_trained_models == num_test_problems:
                raise ValueError(f"Using train problem {c_train_idx} model for test problem: {idx}")
        elif model.model_class == "per_dataset_cluster":
            c_train_idx = []
        else:
            assert model.is_trained, "Model is not trained"
            c_train_idx = ranked_train_idxs[0]
        
        dist_ = data.distances[idx][c_train_idx]
        problem_idx = c_train_idx if model.model_class in ["per_problem", "cluster"] else None
 
        if model.model_class == "per_dataset_cluster":
            cluster_idxs = model.clusters.test_cluster_idxs[idx]
            outputs = model.calculate_metrics(X_test[idx], y_test[idx], difficulties=cluster_idxs)
        else:
            outputs = model.calculate_metrics(X_test[idx], y_test[idx], problem_idx=problem_idx)
        
        if model.model_type == "weak_supervision" and y_test[idx] is not None:
            outputs["class_balance"] = np.mean(y_test[idx])

        outputs["problem"] = sample_idx
        outputs["set"] = "test"
        outputs["close_train_idx"] = problem_idx
        outputs["distance"] = dist_
        outputs["difficulty"] = data.assignments[sample_idx]
        all_test_results.append(outputs)

    all_test_results = pd.DataFrame(all_test_results)
    assert not (all_test_results['top1_positive'].values == np.isnan).any()
    
    return all_results, all_test_results


@hydra.main(config_path="configs", config_name="supervised", version_base=None)
def main(args) -> None:
    if args.get("debug", False):
        args.data_cfg.train_split = 0.2
        args.data_cfg.train_queries = 10 # number of queries to sample from train split
        args.data_cfg.train_samples = 10 # number of samples to sample from train split
        args.data_cfg.same_train_test = True
        args.logging = "none"

    print("W&B config: ", args.wandb_cfg)
    if args.logging == "wandb":
        wandb.init(**args.wandb_cfg, config=OmegaConf.to_container(args, resolve=True))
    train(args)
    if args.logging == "wandb":
        wandb.finish()


def train(args):
    console = Console()
    
    data = VerificationDataset(**args.data_cfg)

    # Print basic dataset info
    console.print(f"\n[bold blue]Dataset Info:[/bold blue]")
    console.print(f"  Train problems: {data.train_data[0].shape[0]}")
    console.print(f"  Train samples per problem: {data.train_data[0].shape[1]}")
    console.print(f"  Test problems: {data.test_data[0].shape[0]}")
    console.print(f"  Test samples per problem: {data.test_data[0].shape[1]}")
    console.print(f"  Verifiers: {len(data.verifier_names)}")
    console.print(f"  Model type: [bold]{args.model_cfg.model_type}[/bold]")

    clusters = None
    if args.model_cfg.model_class == "cluster":
        clusters = ClusteringDataset(**args.model_cfg.cluster_cfg)
        clusters.compute_clusters(data, mode="train")
        num_models = len(clusters.train_clusters)
    elif args.model_cfg.model_class == "per_problem":
        num_models = len(data.train_data[0])
    elif args.model_cfg.model_class == "per_dataset":
        num_models = None
    elif args.model_cfg.model_class == "per_dataset_cluster":
        clusters = ClusteringDataset(**args.model_cfg.cluster_cfg)
        clusters.compute_clusters(data, mode="train")
        num_models = 1
    else:
        raise NotImplementedError(f"Unknown model class: {args.model_cfg.model_class}")

    if args.data_cfg.reward_threshold is not None:
        data.binarize_verifiers(clusters, split="train")
        data.binarize_verifiers(clusters, split="test")

    model = Model(data.verifier_names, clusters, **args.model_cfg, num_models=num_models)

    console.print(f"\n[bold yellow]Training and evaluating model...[/bold yellow]")
    df_train, df_test = train_and_evaluate(data, model, args.fit_cfg)

    # Calculate summary metrics
    try:
        top1_tp_train = df_train["top1_tp"].sum()
        top1_fp_train = df_train["top1_fp"].sum()
        top1_fn_train = df_train["top1_fn"].sum()
        top1_tn_train = df_train["top1_tn"].sum()
        top1_acc_train = (top1_tp_train + top1_tn_train) / (top1_tp_train + top1_tn_train + top1_fp_train + top1_fn_train)

        top1_tp_test = df_test["top1_tp"].sum()
        top1_fp_test = df_test["top1_fp"].sum()
        top1_fn_test = df_test["top1_fn"].sum()
        top1_tn_test = df_test["top1_tn"].sum()
        top1_acc_test = (top1_tp_test + top1_tn_test) / (top1_tp_test + top1_tn_test + top1_fp_test + top1_fn_test)
    except:
        top1_tp_train, top1_fp_train, top1_fn_train, top1_tn_train, top1_acc_train = np.nan, np.nan, np.nan, np.nan, np.nan
        top1_tp_test, top1_fp_test, top1_fn_test, top1_tn_test, top1_acc_test = np.nan, np.nan, np.nan, np.nan, np.nan

    # 1. SUMMARY TABLE
    console.print(f"\n[bold green]SUMMARY RESULTS[/bold green]")
    summary_data = {
        'Set': ['Train', 'Test'],
        'N_Problems': [len(df_train), len(df_test)],
        'Select_Acc': [f"{df_train['top1_positive'].mean():.3f}", f"{df_test['top1_positive'].mean():.3f}"],
        'Sample_Acc': [f"{df_train['sample_accuracy'].mean():.3f}", f"{df_test['sample_accuracy'].mean():.3f}"],
        'Top1_Acc': [f"{top1_acc_train:.3f}", f"{top1_acc_test:.3f}"],
        'TP': [top1_tp_train, top1_tp_test],
        'TN': [top1_tn_train, top1_tn_test], 
        'FP': [top1_fp_train, top1_fp_test],
        'FN': [top1_fn_train, top1_fn_test]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False, tablefmt="grid"))

    # 2. DIFFICULTY BREAKDOWN TABLE
    console.print(f"\n[bold green]RESULTS BY DIFFICULTY LEVEL[/bold green]")
    
    all_train_difficulty_levels = df_train["difficulty"].unique()
    all_test_difficulty_levels = df_test["difficulty"].unique()
    all_difficulty_levels = np.sort(np.unique(np.concatenate([all_train_difficulty_levels, all_test_difficulty_levels])))

    difficulty_results = []
    for difficulty in all_difficulty_levels:
        df_train_diff = df_train[df_train["difficulty"] == difficulty]
        df_test_diff = df_test[df_test["difficulty"] == difficulty]

        train_select_acc = df_train_diff["top1_positive"].mean()
        test_select_acc = df_test_diff["top1_positive"].mean()
        train_sample_acc = df_train_diff["sample_accuracy"].mean()
        test_sample_acc = df_test_diff["sample_accuracy"].mean()
        
        try:
            num_train_problems = len(df_train_diff['top1_tp'])
            train_top1_acc = (df_train_diff["top1_tp"].sum() + df_train_diff["top1_tn"].sum()) / num_train_problems
            num_test_problems = len(df_test_diff['top1_tp'])
            test_top1_acc = (df_test_diff["top1_tp"].sum() + df_test_diff["top1_tn"].sum()) / num_test_problems
        except:
            train_top1_acc = np.nan
            test_top1_acc = np.nan

        difficulty_results.append({
            'Difficulty': int(difficulty),
            'Train_Problems': len(df_train_diff),
            'Test_Problems': len(df_test_diff),
            'Train_Select_Acc': f"{train_select_acc:.3f}",
            'Test_Select_Acc': f"{test_select_acc:.3f}",
            'Train_Sample_Acc': f"{train_sample_acc:.3f}",
            'Test_Sample_Acc': f"{test_sample_acc:.3f}",
            'Train_Top1_Acc': f"{train_top1_acc:.3f}",
            'Test_Top1_Acc': f"{test_top1_acc:.3f}"
        })

    difficulty_df = pd.DataFrame(difficulty_results)
    print(difficulty_df.to_markdown(index=False, tablefmt="grid"))

    # 3. MODEL INFO
    console.print(f"\n[bold cyan]MODEL DETAILS[/bold cyan]")
    console.print(f"  Verifiers used: {len(data.verifier_names)}")
    console.print(f"  Verifier names: {', '.join(data.verifier_names[:5])}{'...' if len(data.verifier_names) > 5 else ''}")

    # Log to wandb
    if wandb.run:
        # Core metrics
        wandb.log({
            "train_select_accuracy": df_train['top1_positive'].mean(),
            "test_select_accuracy": df_test['top1_positive'].mean(),
            "train_sample_accuracy": df_train['sample_accuracy'].mean(),
            "test_sample_accuracy": df_test['sample_accuracy'].mean(),
            "train_top1_accuracy": top1_acc_train,
            "test_top1_accuracy": top1_acc_test,
            "num_verifiers": len(data.verifier_names),
            "verifiers": data.verifier_names
        })
        
        # Per-difficulty metrics
        for _, row in difficulty_df.iterrows():
            difficulty = row['Difficulty']
            wandb.log({
                f"train_select_accuracy_diff_{difficulty}": float(row['Train_Select_Acc']),
                f"test_select_accuracy_diff_{difficulty}": float(row['Test_Select_Acc']),
                f"train_problems_diff_{difficulty}": row['Train_Problems'],
                f"test_problems_diff_{difficulty}": row['Test_Problems'],
            })

        # Save results as artifacts
        run_id = wandb.run.id
        train_file = FIGURES_DIR / run_id / f"df_train.csv"
        test_file = FIGURES_DIR / run_id / f"df_test.csv"

        train_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.parent.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)

        artifact = wandb.Artifact(name="results", type="dataset")
        artifact.add_file(local_path=train_file, name="train")
        artifact.add_file(local_path=test_file, name="test")
        wandb.run.log_artifact(artifact)

    console.print(f"\n[bold green]✅ Training and evaluation complete![/bold green]")
    
    if args.data_cfg.get('save_weaver_scores', False):
        save_weaver_scores_to_dataset(data, model, df_test, args)

    if args.data_cfg.get('save_weaver_model', False):
        save_weaver_model(model, df_test, args)
        
    if wandb.run:
            wandb_url = wandb.run.get_url()
            console.print(f"\n[bold blue]Results logged to W&B:[/bold blue] {wandb_url}")


if __name__ == "__main__":
    main()