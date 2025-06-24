# Selection

This directory contains the code for testing different verifier aggregation methods as described in the Weaver paper. This is where we run the experiments that produce Table 1 and other key results, comparing how well different methods can select correct responses from multiple model generations.

For initial setup and dependencies, see [README.md](../../README.md). To generate datasets for evaluation, see [generation/README.md](../../generation/README.md).

## Core Components

### Main Script (run.py)
The main training/evaluation script that:
1. Loads and splits datasets into train/test
2. Fits aggregation models
3. Evaluates performance.

### Verification Dataset (dataset.py)
Handles dataset loading and preprocessing:
- Loads HuggingFace datasets with verifier scores for each response
- Performs difficulty-aware train/test splitting to ensure balanced evaluation
- Normalizes verifier scores and handles missing values

### Aggregation Methods (models.py)
Contains implementations of Weaver and other baseline selection methods.

##### Weaver (Weak Supervision)
Learns verifier weights without ground truth labels using statistical techniques from weak supervision. Handles verifier binarization, filtering, and dependency modeling. For more information about Weaver, refer to Appendix B of the paper.


##### Verifier-Free Baselines
- **First Sample (Pass@1)**: Use the first generated response
- **Majority Voting**: Select most frequent answer across generations
- **Coverage**: Upper bound - whether any response is correct

##### Alternative Verification Baselines
- **Naive Ensemble**: Simple averaging of all verifier scores  
- **Naive Bayes**: Assumes conditional independence between verifiers
- **Logistic Regression**: Supervised baseline trained on ground truth labels
- **Unsupervised**: EM algorithm with clustering and gating networks

For more information about verification baselines, refer to Appendix C of the paper.

## Understanding Output Metrics

The framework reports several key metrics:

- **Selection Accuracy (`top1_positive`)**: Fraction of problems where selected response is correct
- **Sample Accuracy**: Average accuracy across all generated responses
- **Generation-Verification Gap**: Difference between oracle Pass@K and achieved success rate
- **Per-Difficulty Results**: Performance breakdown by problem difficulty
- **Per-Cluster Results**: Performance for each cluster (when using clustering)

## Reproducing Table 1 & Table 3 Results

The configs for reproducing our main results are in [selection/configs/best_configs](selection/configs/best_configs). These contain the exact hyperparameters used in the paper. The key result to check is test set selection accuracy ("Select_Acc") in the "SUMMARY RESULTS" section that is printed out. Expected runtime of `run.py`: < 1 min.

```bash
# Run experiments from within the selection directory
cd selection

# Weaver results
python run.py --config-path="configs/best_configs" --config-name="MATH-500_70B" # Expected Select_Acc: 93.4%
python run.py --config-path="configs/best_configs" --config-name="GPQA_8B" # Expected Select_Acc: 47.1%
# etc

# Majority voting baseline (Expected Select_Acc: 83.0%)
python run.py --config-name="majority_vote" \
    data_cfg.dataset_name="MATH-500" \
    data_cfg.model_size="70B"

# Override specific parameters
python run.py --config-name="majority_vote" \
    data_cfg.dataset_name="MMLU-Pro" \
    data_cfg.model_size="70B" \
    verifier_cfg.verifier_type="judges" \
    debug=true # runs on a small subset of the dataset
```

## Basic Commands

```bash
# Run experiments from within the selection directory
cd selection

# Weaver on MATH-500 with 70B model
python run.py \
    data_cfg.dataset_name="MATH-500" \
    data_cfg.model_size="70B" \
    model_cfg.model_type="weak_supervision" \
    model_cfg.model_class="per_dataset"

# Naive ensemble baseline
python run.py \
    data_cfg.dataset_name="MATH-500" \
    data_cfg.model_size="70B" \
    model_cfg.model_type="naive_ensemble" \
    model_cfg.model_class="per_dataset"

# Majority voting (verifier-free)
python run.py \
    data_cfg.dataset_name="MATH-500" \
    data_cfg.model_size="70B" \
    model_cfg.model_type="majority_vote" \
    model_cfg.model_class="per_dataset"

# Logistic regression with clustering
python run.py \
    data_cfg.dataset_name="MATH-500" \
    data_cfg.model_size="70B" \
    model_cfg.model_type="logistic_regression" \
    model_cfg.model_class="cluster" \
    model_cfg.cluster_cfg.n_clusters=2 \
    model_cfg.cluster_cfg.cluster_type="unique_extracted_answers"
```

## Experiment Configuration

Our config files are organized into several key sections. Full config files can be found in [configs](configs):

### `verifier_cfg` - Verifier Selection
Controls which verifiers to include in the ensemble:
```yaml
verifier_cfg:
  verifier_type: "all"          # "reward_models", "judges", or "all"
  verifier_size: "all"          # "all", "small", "medium", "large", or integer threshold
  verifier_subset: []           # Optional list of specific verifier names
```

### `data_cfg` - Data Loading and Preprocessing
Configures dataset loading, splitting, and normalization:
```yaml
data_cfg:
  dataset_name: "MATH-500"      # Public dataset name: "MATH-500", "MMLU-Pro", "MMLU", "GPQA", "GPQA-Diamond", "GPQA-1K" (ignored if dataset_path is set)
  model_size: "70B"             # Public dataset model size: "8B", "70B" (ignored if dataset_path is set)
  dataset_path: "/path/to/your/dataset" # Optional: custom dataset path (HF hub or local disk), overrides dataset_name, model_size
  train_split: 0.8              # Fraction for training (1.0 = no test set)
  train_queries: 1              # Fraction/number of train queries to use
  train_samples: 10             # Max samples per problem for training
  test_samples: 10              # Max samples per problem for testing
  reward_threshold: 0.5         # Threshold for binarizing reward model scores
  normalize_type: "all_problems" # "per_problem" or "all_problems"
  normalize_method: "minmax"    # "minmax", "quantile", "winsorize"
  mv_as_verifier: true          # Whether to include majority vote as a verifier
  shuffle_samples: true         # Randomize sample order
  save_weaver_scores: true      # Will save a new column "weaver_scores" to your dataset if model_type == "weak_supervision"
```

### `model_cfg` - Model Architecture and Training
Specifies the aggregation method and training strategy:
```yaml
model_cfg:
  model_type: "weak_supervision"  # Aggregation method
  model_class: "per_dataset"      # Training strategy: "per_problem", "per_dataset", "cluster"
  cluster_cfg:                    # Only needed when model_class="cluster"
    n_clusters: 3                 # Number of clusters
    cluster_type: "by_difficulty" # "by_difficulty", "random", "unique_extracted_answers", "bert_query"
    preserve_ties: true           # Maintain tied difficulty levels in same cluster
```

The framework supports three training strategies via `model_class`:

`model_class: "per_dataset"` (default): Single model trained on all training data, applied to all test problems
- **Best for**: General-purpose verification, limited training data
- **Supports**: All aggregation methods

`model_class: "per_problem"`: Separate model for each test problem using most similar training problems
- **Best for**: When you have sufficient training data and expect problem-specific behavior
- **Supports**: Weaver, Naive Bayes, Logistic Regression, Unsupervised

`model_class: "cluster"`: Groups similar problems and trains specialized model per cluster
- **Best for**: Balancing generalization with specialization
- **Supports**: Weaver, Naive Bayes, Logistic Regression, Unsupervised
- **Requires**: `cluster_cfg` configuration

### `fit_cfg` - Training Configuration
Controls how models are fitted to data:
```yaml
fit_cfg:
  fit_type: "wclosest_to_train"   # "wclosest_to_train" or "search_weights"
```

### `model_params` - Method-Specific Parameters
Each aggregation method has its own parameter section:

**Weaver (weak_supervision):**
```yaml
weak_supervision:
  k: 2                              # Number of classes
  binarize_threshold: 0.5           # Threshold for converting continuous to binary
  drop_imbalanced_verifiers: "adaptive" # Filter strategy: null, "all", "small", "large", "adaptive"
  use_deps: "drop"                  # Dependency handling: "none", "drop", "model"
  drop_k: 3                         # Number of verifiers to keep when dropping
  n_epochs: 1000                    # Training iterations
  lr: 0.00001                       # Learning rate
```

Note:
- `drop_imbalanced_verifiers: "adaptive"` enables Weaver to adaptively choose which verifiers to filter based on problem difficulty.
- `use_deps: "drop"` removes maximally correlated verifiers

**Logistic Regression:**
```yaml
logistic_regression:
  penalty: "l2"                     # Regularization type
  class_weight: "balanced"          # Handle class imbalance
  solver: "newton-cholesky"         # Optimization algorithm
  max_iter: 1000                    # Maximum iterations
```

**Majority Vote:**
```yaml
majority_vote:
  k: 1                              # Number of top answers to consider
  majority_select: "one_sample"     # "majority" or "one_sample"
```

**Naive Bayes:**
```yaml
naive_bayes:
  use_deps: "drop"                  # Dependency handling
  drop_imbalanced_verifiers: "all"  # Verifier filtering
  clip_min: 0.01                    # Lower bound for estimated accuracies
  clip_max: 0.99                    # Upper bound for estimated accuracies
```

### `normalize_method_params` - Normalization Settings
Configure score normalization approaches:
```yaml
normalize_method_params:
  minmax:                             # Min-max scaling to [0,1]
    tmp: # no parameters

  quantile:                           # Quantile-based normalization
    output_distribution: "uniform"    # Target distribution
    n_quantiles: 100                  # Number of quantiles

  winsorize:                          # Clip extreme values
    lower_quantile: 0.05              # Lower percentile to clip
    upper_quantile: 0.95              # Upper percentile to clip
```
