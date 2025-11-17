import numpy as np

import random
from .tensor_decomp import mixture_tensor_decomp_full

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
import itertools
from sklearn.cluster import KMeans
from scipy.special import expit
from scipy.special import logsumexp, expit, softmax

try:
    from metal.label_model import LabelModel
    pass
except ImportError:
    class LabelModel:
        def __init__(self, *args, **kwargs):
            self.dummy = True

from scipy.special import logsumexp

from scipy.stats import norm

from tqdm import tqdm

from scipy.stats import multivariate_normal
from scipy.linalg import solve_triangular


import logging


def calculate_prediction_metrics(probs, y):
    """
    Calculate the accuracy of the most confident prediction across all samples:

    Args:
    - probs: (num_samples, 2)
    - y: (num_samples,)

    Outputs:
        - accuracy of the prediction (selected sample) for each query
        - tp: number of true positives
        - tn: number of true negatives
        - fp: number of false positives
        - fn: number of false negatives
    """   
    # Select sample with highest probability:
    sample_idx, class_idx = np.unravel_index(np.argmax(probs), probs.shape)

    prediction_accuracy = (class_idx == y[sample_idx]).astype(int)

    prediction_tp = int(class_idx == 1 and y[sample_idx] == 1)
    prediction_tn = int(class_idx == 0 and y[sample_idx] == 0)
    prediction_fp = int(class_idx == 1 and y[sample_idx] == 0)
    prediction_fn = int(class_idx == 0 and y[sample_idx] == 1)

    return {
        "prediction_idx": sample_idx,
        "prediction_class": class_idx, # Rename to top1_positive
        "prediction_accuracy": prediction_accuracy,
        "prediction_tp": prediction_tp,
        "prediction_tn": prediction_tn,
        "prediction_fp": prediction_fp,
        "prediction_fn": prediction_fn,
    }


def calculate_top1_metrics(probs, y):
    """
    Top-1 selection based on class-1 probability
    
    Selects the sample with the highest probability for class 1, assumes you're trying to select the one most likely to be positive.
    Then checks if the true label is 1 and updates TP/FP/TN/FN accordingly based on whether the query had any positives.
     
    Args:
        - probs: (num_samples, 2)
        - y: (num_samples,)

    Outputs:
        - accuracy of the prediction (selected sample) for each query
        - top1_positive: whether the top1 prediction is correct
    """
    assert probs.ndim == 2
    assert y.ndim == 1
    assert probs.shape[0] == y.shape[0]

    # Select sample with highest class-1 probability
    top1_idx = np.argmax(probs[: , 1])
    top1_label = y[top1_idx]
    top1_pred = 1  # because we picked the sample with highest class-1 prob

    # Is there any positive in the problem? i.e., is this a "positive" query?
    y_true = np.any(y).astype(int)

    top1_tp = int(top1_label == 1 and y_true == 1)  # picked something and the query had a true answer
    top1_fp = int(top1_label == 1 and y_true == 0)  # picked something in a query with no positives
    top1_tn = int(top1_label == 0 and y_true == 0)  # did NOT pick anything, and none existed
    top1_fn = int(top1_label == 0 and y_true == 1)  # missed something in a query with a true answer

    top1_positive = int(top1_label == top1_pred) # equal to old top1_correct

    return {
        "top1_idx": top1_idx,
        "top1_positive": top1_positive, 
        "has_positive": y_true,
        "top1_tp": top1_tp,
        "top1_fp": top1_fp,
        "top1_tn": top1_tn,
        "top1_fn": top1_fn,
    }


def calculate_sample_metrics(probs, y, return_flat=True):
    """
    Calculate the sample-level metrics for the model:

    Args:
        probs: (num_samples, 2)
        y: (num_samples,)
        return_flat: whether to return the metrics as a sum over all samples

    Returns:
        metrics: (num_samples,)
        metrics["sample_accuracy"]: (num_samples,)
        metrics["sample_tp"]: (num_samples,)
        metrics["sample_tn"]: (num_samples,)
        metrics["sample_fp"]: (num_samples,)
        metrics["sample_fn"]: (num_samples,)
    """ 
    assert probs.ndim == 2
    assert y.ndim == 1
    assert probs.shape[0] == y.shape[0]

    # For each sample, calculate the accuracy of the prediction
    selected_cls_per_sample = np.argmax(probs, axis=1)
    accuracy = (y == selected_cls_per_sample).astype(int)

    # Calculate the TPR and TNR
    tp = (selected_cls_per_sample == 1) & (y == 1)
    tn = (selected_cls_per_sample == 0) & (y == 0)
    fp = (selected_cls_per_sample == 1) & (y == 0)
    fn = (selected_cls_per_sample == 0) & (y == 1)

    if return_flat:
        tp = np.sum(tp)
        tn = np.sum(tn)
        fp = np.sum(fp)
        fn = np.sum(fn)
        accuracy = np.mean(accuracy)

    metrics = {
        "sample_accuracy": accuracy, # sample accuracy
        "sample_tp": tp,
        "sample_tn": tn,
        "sample_fp": fp,
        "sample_fn": fn,
        "num_samples": len(y),
        }

    return metrics


class Model:
    def __init__(self, verifier_names, clusters, model_type, model_class, num_models=None, **kwargs):
        self.verifier_names = verifier_names
        self.model_type = model_type
        self.model_class = model_class
        self.clusters = clusters
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        self.model_params = kwargs.get("model_params", {})
        self.cluster_cfg = kwargs.get("cluster_cfg", {})
        self.logit_adjustment = kwargs.get("logit_adjustment", False)
        self.is_test = False

        if self.model_class == "per_problem":
            assert num_models is not None, "num_problems must be provided for per_problem models."
            self.models = {idx: MODEL_TYPES[self.model_type](verifier_names=self.verifier_names, **self.model_params) for idx in range(num_models)}
            self.is_trained = {idx: False for idx in range(num_models)} 
        elif self.model_class == "per_dataset":
            self.model = MODEL_TYPES[self.model_type](verifier_names=self.verifier_names, **self.model_params)
            self.is_trained = False
        elif self.model_class == "cluster":
            self.models = {idx: MODEL_TYPES[self.model_type](verifier_names=self.verifier_names, **self.model_params) for idx in range(num_models)}
            self.is_trained = {idx: False for idx in range(num_models)}
        elif self.model_class == "per_dataset_cluster":
            self.model = MODEL_TYPES[self.model_type](verifier_names=self.verifier_names, **self.model_params)
            self.is_trained = False
        else:
            raise NotImplementedError(f"Unknown model class: {self.model_class}")

    def get_model(self, group_idx=None):
        if self.model_class == "per_problem":
            assert group_idx is not None, "problem_idx is required for per_problem models."
            return self.models[group_idx]
        elif self.model_class in ["per_dataset", "per_dataset_cluster"]:
            return self.model
        elif self.model_class == "cluster":
            assert group_idx is not None, "problem_idx is required for cluster models."
            return self.models[group_idx]
        else:
            raise NotImplementedError(f"Unknown model class: {self.model_class}")

    def fit(self, X, y, group_idx=None, **kwargs):
        """
        Fit the model on the training data.
        """

        if not(self.model_type in ["majority_vote"]) and X.ndim == 3:
            X = np.vstack(X)
            y = np.concatenate(y)

        model = self.get_model(group_idx)
        model.fit(X, y, **kwargs)
        if group_idx is not None:
            self.is_trained[group_idx] = True
        else:
            self.is_trained = True

        if self.logit_adjustment:
            assert self.model_type == "logistic_regression", "Logit adjustment is only supported for logistic regression."
            # assert model.class_weight is None, "Logit adjustment is not supported for class-weighted models."
            pi_1 = (y == 1).mean()
            pi_0 = 1 - pi_1
            logit_adjustment = np.log(pi_1 / pi_0)
            model.intercept_ -= logit_adjustment
    

    def calculate_metrics(self, X, y, problem_idx=None, **kwargs):
        """
        Calculate the metrics for the model.
        If the model is not trained, return NaN.
        Returns a dictionary with the following keys:
        - top1_correct: whether the model's top1 prediction is correct
        - top1_idx: the index of the top1 prediction
        - acc: the accuracy of the model
        - prediction_accuracy: the accuracy of the model's prediction
        - model_params: the parameters of the model
        - verifier_subset: the subset of verifiers used by the model
        """
        # Get group idx from problem idx
        group_idx = self.problem_idx_to_group_idx(problem_idx)

        # need to be sure that model is fitted
        if problem_idx is not None and not self.is_trained[group_idx]:
            return {
                "top1_positive": np.nan,
                "top1_idx": np.nan,
                "sample_accuracy": np.nan,
                "prediction_accuracy": np.nan,
                "model_params": np.nan,
                "verifier_subset": np.nan
            }
        else:
            if not self.is_trained:
                return {
                    "top1_positive": np.nan,
                    "top1_idx": np.nan,
                    "sample_accuracy": np.nan,
                    "prediction_accuracy": np.nan,
                    "model_params": np.nan,
                    "verifier_subset": np.nan
                }
        
        model = self.get_model(group_idx)
        model.is_test = self.is_test

        metrics = model.calculate_metrics(X, y, **kwargs)
        return metrics
        
    def problem_idx_to_group_idx(self, problem_idx: int) -> int:
        if self.model_class in ["per_dataset", "per_dataset_cluster"]:
            return None 
        elif self.model_class == "per_problem":
            return problem_idx
        elif self.model_class == "cluster":
            for cluster_id, assignments in self.clusters.train_clusters.items():
                if problem_idx in assignments:
                    #print(f"Using model {cluster_id} for problem {problem_idx}.")
                    return cluster_id
            raise ValueError(f"Problem index {problem_idx} not found in any cluster.")
        else:
            raise NotImplementedError(f"Unknown model class: {self.model_class}")


class GatingNetwork:
    def __init__(self, input_dim, num_clusters, lr=0.1):
        self.W = np.random.randn(input_dim, num_clusters) * 0.01
        self.b = np.zeros(num_clusters)
        self.lr = lr

    def predict_logits(self, X_features):
        return X_features @ self.W + self.b  # (Q, K)

    def predict_phi(self, X_features):
        return softmax(self.predict_logits(X_features), axis=1)

    def update(self, X_features, responsibilities):
        # Simple SGD on cross-entropy loss
        phi_pred = self.predict_phi(X_features)
        error = phi_pred - responsibilities  # (Q, K)
        grad_W = X_features.T @ error / X_features.shape[0]
        grad_b = error.mean(axis=0)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

def extract_query_features(X_i):  # X_i: (R, V)
    mean = X_i.mean(axis=0)
    var = X_i.var(axis=0)
    return np.concatenate([mean, var])


class Unsupervised(Model):
    def __init__(self, num_difficulties: int = 3, max_em_iter: int = 100, tol: float = 1e-4, **kwargs):
        """
        Unsupervised model with clustering and logistic regression:
        1. Cluster responses based on verifier assessments
        2. For each cluster, learn a logistic regression model to predict correctness

        Model Assumptions:
            - c_i ~ Categorical(pi)                          # Latent class (cluster) for each query
            - y_ij ∈ {0, 1}                                  # Latent correctness of response j to query i
            - y_ij | c_i ~ Bernoulli(logistic(X_ij^T θ_{c_i}))

        EM Algorithm:
            - E-step: Compute posterior responsibilities p(c_i | X_i)
            - M-step: Fit weighted logistic regressions and update pi

            Inference:
            - p(y_ij = 1 | X_ij) = sum_c p(c_i = c | X_i) * logistic(X_ij^T θ_c)

            Input shape:
            - X: numpy array of shape [Q, R, V], where:
                Q = number of queries
                R = number of responses per query
                V = number of verifiers (features)
        """
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        self.is_test = False
        self.num_clusters = num_difficulties
        self.max_em_iter = max_em_iter
        self.tol = tol
        self.models = [LogisticRegression(
            penalty=kwargs.get("penalty", "l2"),
            class_weight=kwargs.get("class_weight", "balanced"),
            solver=kwargs.get("solver", "lbfgs"),
            max_iter=kwargs.get("max_iter_lr", 1000),
            random_state=kwargs.get("random_state", 42),
            tol=kwargs.get("tol", 1e-4),
            
        ) for _ in range(self.num_clusters)]
        self.phi = np.ones(num_difficulties) / num_difficulties  # Cluster priors
        self.verbose = kwargs.get("verbose", False)
        self.gating_net = None


    def preprocess(self, X):
        assert X.ndim == 2
        return X
    
    def _compute_phi_i(self, X):
        Q, R, V = X.shape
        feats = np.array([extract_query_features(X[i]) for i in range(Q)])
        return self.gating_net.predict_phi(feats)  # shape: (Q, K)

    def _initialize(self, X, y, labels=None):
        Q, R, V = X.shape
        input_dim = 2*V
        self.gating_net = GatingNetwork(input_dim, self.num_clusters)

        if labels is not None:
            for k in range(self.num_clusters):
                mask = labels == k
                if np.sum(mask) == 0:
                    continue
                X_flat = X[mask].reshape(-1, V)
                y_flat = y[mask].reshape(-1)
                self.models[k].fit(X_flat, y_flat)
        else:
            # Compute per-query variance and sort queries by it
            X_query_var = X.var(axis=1).mean(axis=1)  # shape (Q,)
            sorted_indices = np.argsort(X_query_var)
            bucket_sizes = [Q // self.num_clusters + (1 if i < Q % self.num_clusters else 0) for i in range(self.num_clusters)]
            start = 0
            for k in range(self.num_clusters):
                end = start + bucket_sizes[k]
                idx = sorted_indices[start:end]
                if len(idx) == 0:
                    continue
                X_flat = X[idx].reshape(-1, V)
                y_flat = y[idx].reshape(-1)
                self.models[k].fit(X_flat, y_flat)
                start = end

            # Set priors proportional to assigned query counts
            #self.phi = np.array(bucket_sizes, dtype=np.float64)
            #self.phi = self.phi / self.phi.sum()


    def _e_step(self, X, y):
        Q, R, V = X.shape
        responsibilities = np.zeros((Q, self.num_clusters))
        phi_i = self._compute_phi_i(X)  # shape (Q, K)
        for k in range(self.num_clusters):
            logits = X.reshape(Q * R, V) @ self.models[k].coef_.T + self.models[k].intercept_
            probs = expit(logits).reshape(Q, R)
            log_likelihood = y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10)
            responsibilities[:, k] = np.sum(log_likelihood, axis=1) + np.log(phi_i[:, k] + 1e-10)
        log_Z = logsumexp(responsibilities, axis=1, keepdims=True)
        responsibilities = np.exp((responsibilities - log_Z) / self.temperature)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, y, responsibilities):
        Q, R, V = X.shape
        for k in range(self.num_clusters):
            weights = responsibilities[:, k].repeat(R)
            X_flat = X.reshape(Q * R, V)
            y_flat = y.reshape(Q * R)
            self.models[k].fit(X_flat, y_flat, sample_weight=weights)
       # Update gating network parameters
        feats = np.array([extract_query_features(X[i]) for i in range(Q)])
        self.gating_net.update(feats, responsibilities)

    def fit(self, X, y, difficulties=None):
        """
        Fit the model to the data.
        """
        #Q, R, V = X.shape

        X = self.preprocess(X)
        if difficulties.ndim == 0:
            num_queries = 1
        else:
            num_queries = len(difficulties)

        num_responses = X.shape[0] // num_queries
    
        X = X.reshape(num_queries, num_responses, -1)
        y = y.reshape(num_queries, num_responses)

        if not self.is_test:
            # Initialize parameters
            data_fraction = getattr(self, 'deps_data_fraction', 1.0)
            print(f"Initializing parameters with data fraction {data_fraction}", flush=True)
            if data_fraction < 1.0:
                num_samples = len(X)
                num_samples_to_use = int(num_samples * data_fraction)
                # Use the first n_samples_to_use samples
                X_subset = X[:num_samples_to_use]
                y_subset = y[:num_samples_to_use]
                difficulties_subset = difficulties[:num_samples_to_use]
            else:
                X_subset = X
                y_subset = y
                difficulties_subset = difficulties
            self._initialize(X_subset, y_subset, difficulties_subset)
        else:
            print("\nWe don't re-init parameters for test set", flush=True)

        prev_ll = -np.inf
        for iteration in tqdm(range(self.max_em_iter), desc="EM"):
            responsibilities = self._e_step(X, y)
            self._m_step(X, y, responsibilities)
            ll = self._compute_log_likelihood(X, y)
            if np.abs(ll - prev_ll) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            if self.verbose and (iteration + 1) % 1 == 0:
                print(f"Iteration {iteration+1}, Log-likelihood: {ll:.4f}")
                
                # Calculate accuracy using current parameters
                #p_y_given_X = np.sum(p_c_y_given_X, axis=0)  # Marginalize over clusters
                #predicted_labels = (p_y_given_X >= 0.5).astype(int)
                predicted_labels = self.predict(X) 
                if y is not None:
                    accuracy = np.mean(predicted_labels == y)
                    print(f"Current accuracy: {accuracy:.4f}")

            prev_ll = ll

    def _compute_log_likelihood(self, X, y):
        Q, R, V = X.shape
        log_probs = np.zeros((Q, self.num_clusters))
        phi_i = self._compute_phi_i(X)  # shape (Q, K)
        for k in range(self.num_clusters):
            logits = X.reshape(Q * R, V) @ self.models[k].coef_.T + self.models[k].intercept_
            probs = expit(logits).reshape(Q, R)
            log_likelihood = y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10)
            log_probs[:, k] = np.sum(log_likelihood, axis=1) + np.log(phi_i[:, k] + 1e-10)
        return np.sum(logsumexp(log_probs, axis=1))

    def predict_proba(self, X):
        Q, R, V = X.shape
        responsibilities = self._e_step(X, np.zeros((Q, R)))
        probs = np.zeros((Q, R))
        for k in range(self.num_clusters):
            logits = X.reshape(Q * R, V) @ self.models[k].coef_.T + self.models[k].intercept_
            probs_k = expit(logits).reshape(Q, R)
            probs += responsibilities[:, k][:, None] * probs_k
        return probs

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def calculate_metrics(self, X, y, difficulties=None):
        if self.is_test and self.fit_when_calculating_metrics:
            self.fit(X, y, difficulties)

        if difficulties.ndim == 0:
            num_queries = 1 #len(difficulties) if difficulties is not None else 1
            num_responses = X.shape[0] // num_queries
            X = X.reshape(num_queries, num_responses, -1)
            y = y.reshape(num_queries, num_responses)
        elif difficulties.ndim == 1:
            breakpoint()
        else:
            num_queries = len(difficulties)

        assert X.ndim == 3  # (num_queries, num_responses, num_verifiers)

        probs = self.predict_proba(X)  # (num_queries, num_responses)
        y_flat = y.flatten()
        probs_flat = probs.flatten()

        # Format into 2-class probabilities for each response
        probs2d = np.stack([1 - probs_flat, probs_flat], axis=1)

        # Assume helper metric functions are globally available
        outputs1 = calculate_sample_metrics(probs2d, y_flat)
        outputs2 = calculate_top1_metrics(probs2d, y_flat)
        outputs3 = calculate_prediction_metrics(probs2d, y_flat)
        outputs = {**outputs1, **outputs2, **outputs3}
        return outputs


class Unsupervised2(Unsupervised):
    def __init__(self, num_difficulties: int = 3, max_iter: int = 1000, tol: float = 1e-4, **kwargs):
        """
        Implementation of model where:
        - β(d_i) = [β₁(d_i), β₂(d_i), ..., β_V(d_i)]: Parameters depend on difficulty and verifier
        - γ(d_i): Parameter for correctness probability based on difficulty
        - y_ij ~ Bern(σ(γ(d_i))): Correctness follows Bernoulli distribution
        - X_ij ~ N(μ(y_ij, d_i), Σ(y_ij, d_i)): Assessments follow multivariate normal
        - μ_v(y_ij, d_i): Expected assessment from verifier v for response with correctness y_ij and difficulty d_i

        Args:
            num_difficulties: Number of difficulty levels
            num_verifiers: Number of verifiers
            max_iter: Maximum number of iterations for optimization
            tol: Tolerance for convergence
        """
        super().__init__(num_difficulties=num_difficulties, max_iter=max_iter, tol=tol, **kwargs)

        # Parameters to be learned
        self.beta = None  # β_v(d_i) parameters for each difficulty and verifier
        self.gamma = None  # γ(d_i) parameters for each difficulty
        self.mu_correct = None    # μ_v(1, d_i) parameters (mean vectors) for correct responses
        self.mu_incorrect = None  # μ_v(0, d_i) parameters (mean vectors) for incorrect responses
        self.sigma_correct = None    # Σ(1, d_i) parameters (covariance matrices) for correct responses
        self.sigma_incorrect = None  # Σ(0, d_i) parameters (covariance matrices) for incorrect responses
        
        # Initialize parameters for each difficulty level
        self.beta = np.zeros((self.num_difficulties, self.num_verifiers))
        self.gamma = np.zeros(self.num_difficulties)
        self.mu_correct = 0.7*np.ones((self.num_difficulties, self.num_verifiers))
        self.mu_incorrect = 0.3*np.ones((self.num_difficulties, self.num_verifiers))
        self.sigma_correct = np.array([np.eye(self.num_verifiers) for _ in range(self.num_difficulties)])
        self.sigma_incorrect = np.array([np.eye(self.num_verifiers) for _ in range(self.num_difficulties)])

    def initialize_parameters(self, X, y=None, difficulties=None):
        """
        Initialize model parameters.
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            y: Optional array of shape (num_responses,) with true correctness values
        """
        if self.num_verifiers is None:
            self.num_verifiers = X.shape[-1]
            
        # Initialize parameters for each difficulty level
        self.beta = np.zeros((self.num_difficulties, self.num_verifiers))
        self.gamma = np.zeros(self.num_difficulties)
        self.mu_correct = np.zeros((self.num_difficulties, self.num_verifiers))
        self.mu_incorrect = np.zeros((self.num_difficulties, self.num_verifiers))
        self.sigma_correct = np.array([np.eye(self.num_verifiers) for _ in range(self.num_difficulties)])
        self.sigma_incorrect = np.array([np.eye(self.num_verifiers) for _ in range(self.num_difficulties)])
        
        # Initialize parameters based on data
        for diff_level in range(self.num_difficulties):
            diff_mask = (difficulties == diff_level)
            
            # Skip if no data for this difficulty level
            if not np.any(diff_mask):
                # Default values if no data for this difficulty
                self.gamma[diff_level] = 0.0  # 0.5 probability of correctness after sigmoid
                self.mu_correct[diff_level] = 0.7 * np.ones(self.num_verifiers)
                self.mu_incorrect[diff_level] = 0.3 * np.ones(self.num_verifiers)
                continue
                
            X_d = X[diff_mask]
            
            if y is not None:
                # Supervised initialization
                y_d = y[diff_mask]
                correct_mask = (y_d == 1)
                incorrect_mask = (y_d == 0)
                
                # Mean assessment for each verifier for correct responses
                if np.any(correct_mask):
                    self.mu_correct[diff_level] = np.mean(X_d[correct_mask], axis=0)
                    if np.sum(correct_mask) > 1:  # Need at least 2 samples for covariance
                        self.sigma_correct[diff_level] = np.cov(X_d[correct_mask], rowvar=False)
                        # Add small value to diagonal for numerical stability
                        self.sigma_correct[diff_level] += 0.01 * np.eye(self.num_verifiers)
                else:
                    # Default values if no correct samples
                    self.mu_correct[diff_level] = 0.7 * np.ones(self.num_verifiers)
                
                # Mean assessment for each verifier for incorrect responses
                if np.any(incorrect_mask):
                    self.mu_incorrect[diff_level] = np.mean(X_d[incorrect_mask], axis=0)
                    if np.sum(incorrect_mask) > 1:  # Need at least 2 samples for covariance
                        self.sigma_incorrect[diff_level] = np.cov(X_d[incorrect_mask], rowvar=False)
                        # Add small value to diagonal for numerical stability
                        self.sigma_incorrect[diff_level] += 0.01 * np.eye(self.num_verifiers)
                else:
                    # Default values if no incorrect samples
                    self.mu_incorrect[diff_level] = 0.3 * np.ones(self.num_verifiers)
                
                # Initialize gamma using empirical correctness rate
                correctness_rate = np.mean(y_d)

                eps = 1e-6
                correctness_rate = np.clip(correctness_rate, eps, 1 - eps)
                self.gamma[diff_level] = np.log(correctness_rate / (1 - correctness_rate + eps))
            else:
                # Unsupervised initialization - use K-means to cluster into likely correct/incorrect
                from sklearn.cluster import KMeans
                
                # Initialize with K-means (2 clusters)
                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(X_d)
                
                # Determine which cluster is likely "correct" based on mean assessment values
                cluster_means = [np.mean(X_d[clusters == i], axis=0) for i in range(2)]
                correct_cluster = 0 if np.mean(cluster_means[0]) > np.mean(cluster_means[1]) else 1
                incorrect_cluster = 1 - correct_cluster
                
                # Initialize parameters based on clusters
                correct_mask = (clusters == correct_cluster)
                incorrect_mask = (clusters == incorrect_cluster)
                
                # Mean assessment for each verifier for correct responses based on cluster
                self.mu_correct[diff_level] = np.mean(X_d[correct_mask], axis=0)
                
                # Mean assessment for each verifier for incorrect responses based on cluster
                self.mu_incorrect[diff_level] = np.mean(X_d[incorrect_mask], axis=0)
                
                if np.sum(correct_mask) > 1:
                    self.sigma_correct[diff_level] = np.cov(X_d[correct_mask], rowvar=False)
                    self.sigma_correct[diff_level] += 0.01 * np.eye(self.num_verifiers)
                    
                if np.sum(incorrect_mask) > 1:
                    self.sigma_incorrect[diff_level] = np.cov(X_d[incorrect_mask], rowvar=False)
                    self.sigma_incorrect[diff_level] += 0.01 * np.eye(self.num_verifiers)
                
                # Initialize gamma using cluster proportions
                correctness_rate = np.mean(correct_mask)
                self.gamma[diff_level] = np.log(correctness_rate / (1 - correctness_rate + 1e-10))
            
            # Initialize beta parameters for each verifier
            # Beta_v(d_i) represents the discriminative power of verifier v at difficulty d_i
            for v in range(self.num_verifiers):
                self.beta[diff_level, v] = self.mu_correct[diff_level, v] - self.mu_incorrect[diff_level, v]

    def sigmoid(self, x):
        """Apply sigmoid function (σ)"""
        return np.exp(x) / (1 + np.exp(x))
    
    def compute_correctness_probability(self, diff_level):
        """
        Compute p(y_ij = 1) = σ(γ(d_i))
        
        Args:
            diff_level: Difficulty level of the query
            
        Returns:
            Probability of correctness for responses at this difficulty
        """
        return self.sigmoid(self.gamma[diff_level])
    
    def compute_assessment_likelihood(self, X_ij, diff_level, is_correct):
        """
        Compute p(X_ij | y_ij, d_i) using multivariate normal distribution
        
        Args:
            X_ij: Assessment vector of shape (num_verifiers,)
            diff_level: Difficulty level of the query
            is_correct: Boolean indicating if computing for correct (True) or incorrect (False) case
            
        Returns:
            Probability density of the assessment given correctness and difficulty
        """
        if is_correct:
            return multivariate_normal.pdf(
                X_ij, 
                mean=self.mu_correct[diff_level], 
                cov=self.sigma_correct[diff_level]
            )
        else:
            return multivariate_normal.pdf(
                X_ij, 
                mean=self.mu_incorrect[diff_level], 
                cov=self.sigma_incorrect[diff_level]
            )
    
    def compute_log_likelihood(self, X, difficulties, y=None):
        """
        Compute the total log-likelihood of the data given current parameters
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            y: Optional array of shape (num_responses,) with true correctness values
            
        Returns:
            Total log-likelihood value
        """
        log_likelihood = 0.0
        eps = 1e-10
        for diff_level in range(self.num_difficulties):
            diff_mask = (difficulties == diff_level)
            
            # Skip if no data for this difficulty level
            if not np.any(diff_mask):
                continue
                
            X_d = X[diff_mask]
            
            mu1 = self.mu_correct[diff_level]
            mu0 = self.mu_incorrect[diff_level]
            cov1 = self.sigma_correct[diff_level]
            cov0 = self.sigma_incorrect[diff_level]
            prior_correct = self.compute_correctness_probability(diff_level)
            prior_incorrect = 1.0 - prior_correct

            if y is not None:
                y_d = y[diff_mask]

                X_correct = X_d[y_d == 1]
                X_incorrect = X_d[y_d == 0]

                if X_correct.shape[0] > 0:
                    log_likelihood += np.sum(
                        multivariate_normal_pdf_fast(X_correct, mean=mu1, cov=cov1 + eps * np.eye(cov1.shape[0]))
                    )

                if X_incorrect.shape[0] > 0:
                    log_likelihood += np.sum(
                        multivariate_normal_pdf_fast(X_incorrect, mean=mu0, cov=cov0 + eps * np.eye(cov0.shape[0]))
                    )

            else:
                # logpdf for each point under both classes
                log_lik_y1 = np.log(multivariate_normal_pdf_fast(X_d, mean=mu1, cov=cov1 + eps * np.eye(cov1.shape[0])))
                log_lik_y0 = np.log(multivariate_normal_pdf_fast(X_d, mean=mu0, cov=cov0 + eps * np.eye(cov0.shape[0])))

                log_joint_y1 = log_lik_y1 + np.log(prior_correct + eps)
                log_joint_y0 = log_lik_y0 + np.log(prior_incorrect + eps)

                # log-sum-exp for marginal log-likelihood
                log_marginal = logsumexp(np.stack([log_joint_y1, log_joint_y0], axis=0), axis=0)
                log_likelihood += np.sum(log_marginal)

        return log_likelihood
    
    def e_step(self, X, difficulties):
        """
        E-step of EM algorithm: compute posterior probabilities of correctness
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            
        Returns:
            Array of same shape as difficulties with posterior probabilities P(y_ij=1|X_ij,d_i)
        """
        posterior_probs = np.zeros(len(X))
        
        for diff_level in range(self.num_difficulties):
            diff_mask = (difficulties == diff_level)
            
            # Skip if no data for this difficulty level
            if not np.any(diff_mask):
                continue
                
            X_d = X[diff_mask]
            
            # Get prior probability for this difficulty level
            prior_correct = self.compute_correctness_probability(diff_level)
            prior_incorrect = 1 - prior_correct

            likelihood_correct = multivariate_normal_pdf_fast(
                X_d, 
                mean=self.mu_correct[diff_level], 
                cov=self.sigma_correct[diff_level]
            )
            
            likelihood_incorrect = multivariate_normal_pdf_fast(
                X_d, 
                mean=self.mu_incorrect[diff_level], 
                cov=self.sigma_incorrect[diff_level]
            )
            
            # Compute posteriors
            joint_correct = likelihood_correct * prior_correct
            joint_incorrect = likelihood_incorrect * prior_incorrect
            marginal = joint_correct + joint_incorrect
            posteriors = joint_correct / (marginal + 1e-10)
            
            # Store results
            posterior_probs[diff_mask] = posteriors

        return posterior_probs
    
    def m_step(self, X, difficulties, posterior_probs):
        """
        M-step of EM algorithm: update parameters to maximize expected log-likelihood
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            posterior_probs: Array of same shape as difficulties with posterior probabilities P(y_ij=1|X_ij,d_i)
        """
        for diff_level in range(self.num_difficulties):
            diff_mask = (difficulties == diff_level)
            
            # Skip if no data for this difficulty level
            if not np.any(diff_mask):
                continue
                
            X_d = X[diff_mask]
            post_d = posterior_probs[diff_mask]
            
            # Weights for correct and incorrect responses
            weights_correct = post_d
            weights_incorrect = 1 - post_d
            
            # Total weights for normalization
            total_correct = np.sum(weights_correct)
            total_incorrect = np.sum(weights_incorrect)
            
            # Update gamma (prior probability parameter)
            mean_posterior = np.mean(post_d)
            self.gamma[diff_level] = np.log(mean_posterior / (1 - mean_posterior + 1e-10))
            
            # Update mu_correct for each verifier (weighted mean for correct responses)
            if total_correct > 0:
                self.mu_correct[diff_level] = np.dot(weights_correct, X_d) / total_correct
            
            # Update mu_incorrect for each verifier (weighted mean for incorrect responses)
            if total_incorrect > 0:
                self.mu_incorrect[diff_level] = np.dot(weights_incorrect, X_d) / total_incorrect
            
            # Update sigma_correct (weighted covariance for correct responses)
            if total_correct > 1:  # Need at least 2 effective samples
                # Center data
                centered_d = X_d - self.mu_correct[diff_level]
                # Use matrix multiplication for weighted covariance
                weighted_centered = np.sqrt(weights_correct.reshape(-1, 1)) * centered_d
                # The outer product gives the covariance matrix
                self.sigma_correct[diff_level] = np.dot(weighted_centered.T, weighted_centered) / total_correct
                # Add small value to diagonal for numerical stability
                self.sigma_correct[diff_level] += 0.01 * np.eye(self.sigma_correct[diff_level].shape[0])
                            
            # Update sigma_incorrect (weighted covariance for incorrect responses)
            if total_incorrect > 1:  # Need at least 2 effective samples
                centered_d = X_d - self.mu_incorrect[diff_level]
                weighted_centered = np.sqrt(weights_incorrect.reshape(-1, 1)) * centered_d
                self.sigma_incorrect[diff_level] = np.dot(weighted_centered.T, weighted_centered) / total_incorrect
                self.sigma_incorrect[diff_level] += 0.01 * np.eye(self.sigma_incorrect[diff_level].shape[0])
            
            #Update beta parameters individually for each verifier
            self.beta[diff_level] = self.mu_correct[diff_level] - self.mu_incorrect[diff_level]

    
    def fit(self, X, y=None, difficulties=None, verbose=True):
        """
        Fit the model parameters using EM algorithm
        
        Args:
            X: Array of shape (num_queries*num_responses, num_verifiers) with verifier assessments
            y: Optional array of shape (num_queries*num_responses,) with true correctness values
            difficulties: Array of shape (num_queries,) with difficulty levels for each response
            verbose: Whether to print progress
            
        Returns:
            List of log-likelihood values during training
        """
        X = self.preprocess(X)

        num_queries = len(difficulties)
        num_responses = X.shape[0] // num_queries

        difficulties = np.repeat(difficulties, num_responses)
                
        if not self.is_test:
            # Initialize parameters
            data_fraction = getattr(self, 'deps_data_fraction', 1.0)
            print(f"Initializing parameters with data fraction {data_fraction}", flush=True)
            if data_fraction < 1.0:
                num_samples = len(X)
                num_samples_to_use = int(num_samples * data_fraction)
                # Use the first n_samples_to_use samples
                X_subset = X[:num_samples_to_use]
                y_subset = y[:num_samples_to_use]
                difficulties_subset = difficulties[:num_samples_to_use]
            else:
                X_subset = X
                y_subset = y
                difficulties_subset = difficulties
            self.initialize_parameters(X_subset, y_subset, difficulties_subset)

        else:
            print("\nWe dont' re-init parameters for test set", flush=True)

        # For unsupervised learning, use EM algorithm
        log_likelihood_old = -np.inf
        log_likelihoods = []
        
        # EM iterations
        for iteration in tqdm(range(self.max_iter), desc="EM iterations"):
            # E-step: compute posterior probabilities
            posterior_probs = self.e_step(X, difficulties)
 
            # M-step: update parameters
            self.m_step(X, difficulties, posterior_probs)
            
            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(X, difficulties)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                if verbose:
                    print(f"Converged after {iteration+1} iterations with tolerance {self.tol}")
                break
                
            log_likelihood_old = log_likelihood
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, Log-likelihood: {log_likelihood:.4f}")
                
                # Calculate accuracy using current parameters
                predicted_labels = (posterior_probs >= 0.5).astype(int)
                if y is not None:
                    accuracy = np.mean(predicted_labels == y)
                    print(f"Current accuracy: {accuracy:.4f}")
        
    
    def predict_correctness(self, X, difficulties):
        """
        Predict correctness probabilities for responses
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            
        Returns:
            Array of correctness probabilities for each response
        """
        # Use E-step to compute posterior probabilities
        return self.e_step(X, difficulties)
    
    def calculate_metrics(self, X, y, difficulties):
        """
        Calculate performance metrics
        
        Args:
            X: Array of shape (num_responses, num_verifiers) with verifier assessments
            y: Array of shape (num_responses,) with true correctness values
            difficulties: Array of shape (num_responses,) with difficulty levels for each response
            
        Returns:
            Dictionary of metrics
        """
        if self.is_test and self.fit_when_calculating_metrics:
            self.fit(X, y, difficulties)

        X = self.preprocess(X)

        if np.ndim(difficulties) == 0:
            num_responses = X.shape[0]
            difficulties = np.repeat(difficulties, num_responses)
        
        # Predict correctness probabilities
        predicted_probs = self.predict_correctness(X, difficulties) # (num_responses, num_verifiers)
        probs = np.asarray([1-predicted_probs, predicted_probs])  # (num_responses, 2, num_verifiers)
        probs = np.moveaxis(probs, 0, -1)

        outputs1 = calculate_sample_metrics(probs, y)
        outputs2 = calculate_top1_metrics(probs, y)
        outputs3 = calculate_prediction_metrics(probs, y)
        outputs = {**outputs1, **outputs2, **outputs3}
        return outputs


class WeakSupervised(LabelModel):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        super().__init__(k=self.k, seed=self.seed)
        self.is_test = False
        self.use_continuous = kwargs.get("use_continuous", False)
        self.use_tensor_decomp = kwargs.get("use_tensor_decomp", False)
        self.random_seed = kwargs.get("seed", 123)
        self.fit_when_calculating_metrics = kwargs.get("fit_when_calculating_metrics", False)
        self.minimum_density = 0
        self.drop_k = kwargs.get("drop_k", 3)  # Default to 3 if not specified
        self.drop_at_test = kwargs.get("drop_at_test", True)
        # when True: if we want to drop all verifiers, drop none:
        self.drop_imbalanced_fallback = kwargs.get("drop_imbalanced_fallback", False)

    def _get_deps(self, votes, truth, density=0.1):
        # assumed  num_samples x num_verifiers
        # For now, assume we have access to the true covariance matrix --- stack both votes and labels
        
        # Use only a fraction of the data for dependency modeling if specified
        data_fraction = getattr(self, 'deps_data_fraction', 1.0)
        if data_fraction < 1.0:
            print(f"Using first {data_fraction} fraction of data for dependency modeling.", flush=True)
            n_samples = len(votes)
            n_samples_to_use = int(n_samples * data_fraction)
            # Use the first n_samples_to_use samples
            votes = votes[:n_samples_to_use]
            truth = truth[:n_samples_to_use]
            
        all_scores = np.hstack([votes, truth[:, np.newaxis]])
        cov = np.cov(all_scores.T)
        cov = cov + 1e-6 * np.eye(cov.shape[0]) # add small value to diagonal to make it invertible
        inv_cov = np.linalg.inv(cov)
        # remove the last row/column
        inv_cov = inv_cov[:, :-1]
        inv_cov = inv_cov[:-1, :]
        
        m = inv_cov.shape[0]

        if density < 1:
            k = int(density * (len(inv_cov) * (len(inv_cov) - 1) // 2)) # desired density 
        else:
            k = density
        deps = []
        sorted_idxs = np.argsort(-np.abs(inv_cov), axis=None)
        for idx in sorted_idxs:
            i = int(np.floor(idx / m))
            j = idx % m 
            if (j, i) in deps or i == j:
                continue
            deps.append((i, j))
            if len(deps) == k:
                break

        return deps 
    
    def _drop_deps(self, votes, truth, current_verifiers, k=None):
        """
            Select the top k maximally independent verifiers based on the inverse covariance matrix of the scores.
            Same function as in WS.
        """
        # Use only a fraction of the data for dependency modeling if specified
        data_fraction = getattr(self, 'deps_data_fraction', 1.0)
        if data_fraction < 1.0:
            print(f"Using first {data_fraction} fraction of data for dependency modeling.", flush=True)
            n_samples = len(votes)
            n_samples_to_use = int(n_samples * data_fraction)

            if n_samples_to_use == 1:
                return votes, np.arange(votes.shape[-1])

            # Use the first n_samples_to_use samples
            votes = votes[:n_samples_to_use]
            truth = truth[:n_samples_to_use]

        # Use drop_k from config if k is not provided
        k = self.drop_k if k is None else k
        k = min(k, len(current_verifiers))
        print(f"Finding maximally independent verifier set of size {k}", flush=True)
        n_verifiers = votes.shape[-1]
        triple_to_marginal = {}
        triple_to_sparsity = {}
        for triple in itertools.combinations(range(n_verifiers), k):
            triple = list(triple)
            
            # compute inverse covariance matrix on the selected verifiers + truth 
            selected_scores = np.hstack([votes[:, triple], truth[:, np.newaxis]])
            selected_cov = np.cov(selected_scores.T)

            try:
                selected_inv_cov = np.linalg.inv(selected_cov)
            except np.linalg.LinAlgError:
                selected_cov = selected_cov + 1e-6 * np.eye(selected_cov.shape[0]) # add small value to diagonal to make it invertible
                selected_inv_cov = np.linalg.inv(selected_cov)

            # discard the covariance with the true answer
            selected_inv_cov = selected_inv_cov[:, :-1]
            selected_inv_cov = selected_inv_cov[:-1, :]

            # set diagonal to 0 (we don't count dependencies with itself)
            np.fill_diagonal(selected_inv_cov, 0)
            
            density = np.abs(selected_inv_cov).sum()

            # record largest magnitude element 
            s = np.abs(selected_inv_cov).max()
            triple_to_sparsity[tuple(triple)] = s
            
            marginals = votes[:, triple].mean(axis=0)
            triple_to_marginal[tuple(triple)] = marginals

        # Use stable sort to ensure deterministic results
        sorted_sparsity = {k: v for k, v in sorted(triple_to_sparsity.items(), key=lambda x: (x[1], str(x[0])))}

        top_triple = list(sorted_sparsity.keys())[0]
        triple_names = [v for i, v in enumerate(current_verifiers) if i in top_triple]
        print(f"Top triple: {triple_names}, sparsity: {sorted_sparsity[top_triple]}", flush=True)
        if any(triple_to_marginal[top_triple] > 0.9) or any(triple_to_marginal[top_triple] < 0.1):
            print(f"WARNING: Some of the verifiers in the top triple have marginal probabilities that are too extreme: {triple_names}, {triple_to_marginal[top_triple]}", flush=True)
            
        
        # Add these lines to calculate and print the density of the selected triple
        selected_scores = np.hstack([votes[:, top_triple], truth[:, np.newaxis]])
        selected_cov = np.cov(selected_scores.T)
        try:
            selected_inv_cov = np.linalg.inv(selected_cov)
        except np.linalg.LinAlgError:
            selected_cov = selected_cov + 1e-6 * np.eye(selected_cov.shape[0])
            selected_inv_cov = np.linalg.inv(selected_cov)
        selected_inv_cov = selected_inv_cov[:, :-1]
        selected_inv_cov = selected_inv_cov[:-1, :]
        np.fill_diagonal(selected_inv_cov, 0)
        selected_density = np.abs(selected_inv_cov).sum()
        print(f"Selected Triple Density: {selected_density}", flush=True)
        
        top_triple = np.array(list(top_triple))
        votes = votes[:, top_triple]
        return votes, top_triple

    def preprocess(self, X):
        assert X.ndim ==2
        if self.use_continuous == False:
            assert len(np.unique(X)) in [1, 2], "X should be binarized, did you set reward_threshold?"
        return X

    
    def _get_ws_estimated_verifier_class_accuracies(self, n_verifiers):
        """
            After computing the label model, we get each verifier's class-conditional accuracy, estimated using WS:
            Pr(lf_i = y | y = 1) and Pr(lf_i = y | y = 0)

            This is useful in comparing WS's estimated TPR/FPR against the true TPR/FPR. 

            Args:
            - label_model: trained WS label model.
            - n_verifiers: number of verifiers 

            Returns:
            - (n_verifiers, n_classes) accuracy matrix A where A[i, j] = Pr(lf_i = y | y = j).
        """

        weights = self.get_conditional_probs().reshape((n_verifiers, -1, self.k))

        weights = weights[:, 1:, :]  # This keeps only the last two rows of the 3-row dimension

        TNR = np.array([matrix[0, 0] for matrix in weights])
        TPR = np.array([matrix[1, 1] for matrix in weights])
        FPR = np.array([matrix[1, 0] for matrix in weights]) 
        FNR = np.array([matrix[0, 1] for matrix in weights])

        return TPR, TNR, FPR, FNR


    def _get_balanced_idxs(self, marginals, rule):
        """
            Get the indices of the verifier scores that are balanced:
            rule: "all", "small", "large"

            all : any verifier with p(y) > 0.1 and p(y) < 0.9
            small: any verifier with p(y) > 0.5, i.e. predicts mostly 'yes'
            large: any verifier with p(y) < 0.5, i.e. predicts mostly 'no'
        """
        if rule == "all":
            balanced_idxs = np.where((marginals > 0.1) & (marginals < 0.9))[0]
        elif rule == "small":
            balanced_idxs = np.where(marginals > 0.5)[0]
        elif rule == "large":
            balanced_idxs = np.where(marginals < 0.5)[0]
        else: # Use all indices if not specified
            balanced_idxs = np.arange(len(marginals))

        # Fallback: if filtering removes everything, use all verifier
        if getattr(self, "drop_imbalanced_fallback", False) and len(balanced_idxs) == 0:
            print(f"WARNING: Rule '{rule}' dropped all verifiers — falling back to using all.", flush=True)
            balanced_idxs = np.arange(len(marginals))

        return balanced_idxs

    def fit(self, X, y):

        X = self.preprocess(X)

        # When on the training set or if we are using labels on the test set:
        if (not self.is_test) or (self.use_label_on_test):
            self.verifier_idxs = np.arange(X.shape[1])

            # if we are testing, and we don't want to use labels 
            # then fix to float

            # Get data fraction for both class balance and verifier dropping
            data_fraction = getattr(self, 'deps_data_fraction', 1.0)
            if data_fraction < 1.0:
                print(f"Using first {data_fraction} fraction of data for dependency modeling and verifier dropping.", flush=True)
                n_samples = len(y)
                n_samples_to_use = int(n_samples * data_fraction)
                # Use subset of data for all label-dependent operations
                X_subset = X[:n_samples_to_use]
                y_subset = y[:n_samples_to_use]
            else:
                X_subset = X
                y_subset = y
        else:
            X_subset = X

        # Drop imbalanced verifiers if specified: 
        # Uses the test set when adaptive= True
        if self.drop_imbalanced_verifiers is not None and (not self.is_test or self.drop_at_test):
            if self.drop_imbalanced_verifiers == "adaptive":
                assert self.use_label_on_test, "Adaptive drop requires use_label_on_test to be True"
                cb = y_subset.mean()  # Use subset for class balance calculation
                if cb < 0.2:
                  rule = "large" # discard all the verifiers that vote mostly 'yes'
                elif cb > 0.8:
                    rule = "small" # discard all the verifiers that vote mostly 'no' if the cluster is easy 
                else:
                    rule = "all"
                print(f"Setting rule for discarding verifiers to be: {rule}.", flush=True)
            else:
                rule = self.drop_imbalanced_verifiers

            marginals = X.mean(axis=0)  # Use all data for marginals calculation
            balanced_idxs = self._get_balanced_idxs(marginals, rule)
            discarded_names = [v for i, v in enumerate(self.verifier_names) if i not in balanced_idxs]
            balanced_names = [v for i, v in enumerate(self.verifier_names) if i in balanced_idxs]
            print(f"Discarding {len(discarded_names)}/{X.shape[1]} verifiers: {discarded_names}", flush=True)
            X = X[:, balanced_idxs]  # Apply to full X
            self.verifier_idxs = balanced_idxs
        else:
            # At test time, we do not drop any verifiers:
            balanced_idxs = self.verifier_idxs
            balanced_names = [v for i, v in enumerate(self.verifier_names) if i in balanced_idxs]
            X = X[:, balanced_idxs]  # Apply to full X


        # Get the class balance:
        # When on the training set or if we are using labels on the test set:
        if (not self.is_test) or self.use_label_on_test:
            cb_args = self.cb_args.class_balance 
        else:
            cb_args = 0.5

        if type(cb_args) == str and cb_args == "labels":
            mean_correctness = y_subset.mean()  # Use subset for class balance calculation
            class_balance = np.asarray([1- mean_correctness, mean_correctness])
        elif type(cb_args) == float:
            class_balance = np.asarray([1- cb_args, cb_args])
        else:
            raise ValueError(f"Unknown class balance: {cb_args}")
        
        if self.use_tensor_decomp:
            print("Estimating class balance using tensor decomposition method.")
            X_transpose = X.T
            y_emb_unique = np.eye(2)[[np.arange(2)]].squeeze()
            # construct label matrix embeddings
            L_emb =[[y_emb_unique[l,:] for l in L_i] for L_i in X_transpose]
            L_emb = np.array(L_emb)

            # take any three verifiers' outputs for a tensor
            n_lfs = list(range(X.shape[1]))
            three_combos = np.array(list(itertools.combinations(n_lfs, 3)))
            np.random.seed(self.random_seed)  # Set numpy random seed for tensor decomposition
            random.seed(self.random_seed)  # Set random seed for sampling
            random_elements = random.sample(range(len(three_combos)), 30)

            # use each tensor to decompose and collect their recovered weights as class balance
            w_rec_collection = []
            for _, combo in tqdm(enumerate(three_combos[random_elements], 1), total=len(three_combos[random_elements])):
            
                (w_rec, mu_hat_1, mu_hat_2, mu_hat_3,) = mixture_tensor_decomp_full(
                    w = np.ones(X.shape[0]) / X.shape[0],
                    x1 = L_emb[combo[0], :, :].T,
                    x2 = L_emb[combo[1], :, :].T,
                    x3 = L_emb[combo[2], :, :].T,
                    k = 2,
                    debug = False
                )
                w_rec_collection.append(w_rec)

            # take the mean of estimated recovered weights as final class balance estimation
            w_rec_collection = np.array(w_rec_collection)
            class_balance = np.mean(w_rec_collection, axis=0)

        cb_inputs ={"class_balance": class_balance}

        # If we are on the training set or if we are using labels on the test set:
        if (not self.is_test) or (self.use_label_on_test):
            if self.use_deps == "model":
                deps = self._get_deps(X, y, density=getattr(self, 'deps_density', 0.1))
                print(f"Modeling {len(deps)} deps: {deps}", flush=True)
            elif self.use_deps == "drop":
                _, remaining_idxs = self._drop_deps(X, y, current_verifiers=balanced_names if self.drop_imbalanced_verifiers is not None else self.verifier_names)
                self.verifier_idxs = self.verifier_idxs[remaining_idxs]
                X = X[..., remaining_idxs]
                print(f"verifier_idxs: {self.verifier_idxs}", flush=True)

        if self.use_continuous == False:
            self.train_model(
                X+1, # binary votes (shifted by 1) 
                deps = deps if self.use_deps == 'model' else [], # for now, no dependencies
                L_train_continuous=None,
                abstains=False, 
                symmetric=False, 
                n_epochs=self.n_epochs, 
                mu_epochs = self.mu_epochs,
                log_train_every=self.log_train_every,
                lr=self.lr,
                **cb_inputs, # the class balance Pr(y = 1) is either set to a hardcoded value or computed from labeled data
            )
            # print("mu:", self.mu.shape, self.mu)

            n_verifiers = X.shape[-1]
            TPR, TNR, FPR, FNR = self._get_ws_estimated_verifier_class_accuracies(n_verifiers)
            # print(f"WS TPR: {TPR}, WS TNR: {TNR}", flush=True)
        else:
            print("Using Continuous Label Model", flush=True)
            L_train = X
            if (not self.is_test) or (self.use_label_on_test):
                # Using y labels for variance estimation
                self.var_y = np.var(y_subset)
            # Actually we don't need it, we can use X for estimation. Performance doesn't drop.
            self.var_y = np.mean(np.var(L_train, axis=1))
            self.n, self.m = L_train.shape
            # Compute the covariance matrix of the verifiers
            self.O = np.transpose(L_train).dot(L_train) / self.n
            # Compute the covariance matrix of the verifiers and the true labels
            self.Sigma_hat = np.zeros([self.m+1, self.m+1])
            # Compute the covariance matrix of the verifiers
            self.Sigma_hat[:self.m, :self.m] = self.O

            # Init dict to collect accuracies in triplets
            acc_collection = {}
            for i in range(self.m):
                acc_collection[i] = []

            # Collect triplet results: 
            # For each verifier, compute the accuracy of the triplet of verifiers
            if self.m >= 3:
                for i in range(self.m):
                    for j in range(i+1, self.m):
                        for k in range(j+1, self.m):
                            acc_i = np.sqrt(self.O[i, j] * self.O[i, k] * self.var_y / self.O[j, k])
                            acc_j = np.sqrt(self.O[j, i] * self.O[j, k] * self.var_y / self.O[i, k])
                            acc_k = np.sqrt(self.O[k, i] * self.O[k, j] * self.var_y / self.O[i, j])
                            acc_collection[i].append(acc_i)
                            acc_collection[j].append(acc_j)
                            acc_collection[k].append(acc_k)

                # Take average
                for i in range(self.m):                
                    self.Sigma_hat[i, self.m] = np.average(acc_collection[i])
                    self.Sigma_hat[self.m, i] = np.average(acc_collection[i])

            elif self.m == 2:
                # Fallback: Use pairwise agreement as proxy
                acc_0 = np.sqrt(self.O[0, 1] * self.var_y / (self.O[1, 1] + 1e-8))
                acc_1 = np.sqrt(self.O[1, 0] * self.var_y / (self.O[0, 0] + 1e-8))
                self.Sigma_hat[0, self.m] = acc_0
                self.Sigma_hat[1, self.m] = acc_1
                self.Sigma_hat[self.m, 0] = acc_0
                self.Sigma_hat[self.m, 1] = acc_1
            
            elif self.m == 1:
                # Fallback: Use diagonal O directly
                # The continuous model estimates verifier reliabily by inverting or normalizing the covariance matrix
                self.Sigma_hat[0, self.m] =  np.mean(L_train[:, 0]) #acc
                self.Sigma_hat[self.m, 0] = np.mean(L_train[:, 0])
                
                
            # Fill in the last diagonal element
            self.Sigma_hat[self.m, self.m] = self.var_y
   
        logging.info(f"Data used in fit: Num samples {X.shape[0]}, Num verifiers {X.shape[1]}")
        logging.info(f"Verifiers used in fit: N={len(self.verifier_idxs)}: {[self.verifier_names[i] for i in self.verifier_idxs]}")
        
        print(80 * "-")
        print(f"{'Verifier Name':<50} | {'P(S=0|Y=0)':<10} | {'P(S=1|Y=1)':<10}")
        print(80 * "-")
        for i, v_idx in enumerate(self.verifier_idxs):
            verifier_name = self.verifier_names[v_idx]
            if verifier_name.startswith('VerifierType'):
                verifier_name = self.verifier_names[v_idx].split('.', 1)[1]
            if verifier_name.startswith('~VerifierType'):
                verifier_name = "~" + self.verifier_names[v_idx].split('.', 1)[1]
            p_s0_y0 = self.mu[2 * i, 0]
            p_s1_y1 = self.mu[2 * i + 1, 1]
            print(f"{verifier_name:<50} | {p_s0_y0:<10.2f} | {p_s1_y1:<10.2f}")
        print(80 * "-")

    def predict_proba(self, X):
        X = X[:, self.verifier_idxs] # use only the verifiers that were used to fit the model
        if self.use_continuous:
            probs = np.zeros((X.shape[0], 2))
            for i in range(X.shape[0]):
                prob = np.expand_dims(self.Sigma_hat[self.m, :self.m], axis=0) \
                                    .dot(np.linalg.inv(self.Sigma_hat[:self.m, :self.m])) \
                                    .dot(np.expand_dims(X[i, :self.m], axis=1))
                probs[i, 0] = 1 - prob.item()
                probs[i, 1] = prob.item()
            return probs
        else:
            return super().predict_proba(X)

    def calculate_metrics(self, X, y):
        """
            X: (n_generations, n_verifiers) for a single problem
            y: (n_generations,) for a single problem
        """
        
        if self.is_test and self.fit_when_calculating_metrics:
            print(f"Fitting when calculating metrics:", flush=True)
            self.fit(X, y)   

        X = self.preprocess(X) # even if fit(X, y) preprocesses it, that does not change X
        if self.metric == "scores":
            probs = self.predict_proba(X+1)
        else:
            raise NotImplementedError(f"Unknown metric: {self.metric}")

        outputs1 = calculate_sample_metrics(probs, y)
        outputs2 = calculate_top1_metrics(probs, y)
        outputs3 = calculate_prediction_metrics(probs, y)
        outputs = {**outputs1, **outputs2, **outputs3}

        if self.use_continuous == False:
            outputs["model_params"] = self._get_ws_estimated_verifier_class_accuracies(len(self.verifier_idxs))
        else:
            outputs["model_params"] = None
        outputs["verifier_subset"] = [v for i, v in enumerate(self.verifier_names) if i in self.verifier_idxs]

        # Class 1 probabilities (weaver scores for each generation)
        outputs["weaver_sample_scores"] = probs[:, 1].tolist()

        return outputs
    
    def get_verifier_ranking(self, X, y=None):
        """
            Get the ranking of the verifiers based on their F1 score.
        """
        if self.use_continuous:
            X = X[:, self.verifier_idxs] # use only the verifiers that were used to fit the model
            # Extract the covariances of the verifiers with the true label
            covariances = self.Sigma_hat[:self.m, self.m]
            # Rank verifiers by the magnitude of this covariance
            ranking = np.argsort(-np.abs(covariances))  # Rank by |Cov(X_i, Y)|
            # Return from most to least informative:
            # Large positive covariances mean that the verifier agrees with the true label
            # Large negative covariances mean that the verifier disagrees with the true label
            # covariance near 0 -> verifier is uninformative or noisy
            return ranking
        else:
            # For the discrete case,:
            # Unsupervised: Use WS-estimated verifier reliabilities
            TPR, TNR, FPR, FNR = self._get_ws_estimated_verifier_class_accuracies(len(self.verifier_idxs))
            balanced_accuracy = 0.5 * (TPR + TNR)
            # sort verifiers by balanced accuracy
            ranking = np.argsort(-balanced_accuracy)
            return ranking


class LogisticRegression2(LogisticRegression):
    def __init__(self, **kwargs):
        # remove verifier names from kwargs
        self.verifier_names = kwargs.pop("verifier_names", None)
        super().__init__(**kwargs)

    def calculate_metrics(self, X, y):
        probs = self.predict_proba(X)

        outputs1 = calculate_sample_metrics(probs, y)
        outputs2 = calculate_top1_metrics(probs, y)
        outputs3 = calculate_prediction_metrics(probs, y)
        outputs = {**outputs1, **outputs2, **outputs3}

        outputs["model_params"] = None
        outputs["verifier_subset"] = None

        return outputs


class NaiveBayes(Model):
    """
    Naive Bayes:
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set additional attributes dynamically

    def preprocess(self, X):
        assert X.ndim ==2
        assert len(np.unique(X)) == 2, "X should be binarized, did you set reward_threshold?"
        return X
    
    def _drop_deps(self, votes, truth, current_verifiers, k=3):
        """
            Select the top k maximally independent verifiers based on the inverse covariance matrix of the scores.
            Same function as in WS.
        """
        k = min(k, len(current_verifiers))
        print(f"Finding maximally independent verifier set of size {k}", flush=True)
        n_verifiers = votes.shape[-1]
        triple_to_marginal = {}
        triple_to_sparsity = {}
        for triple in itertools.combinations(range(n_verifiers), k):
            triple = list(triple)
            
            # compute inverse covariance matrix on the selected verifiers + truth 
            selected_scores = np.hstack([votes[:, triple], truth[:, np.newaxis]])
            selected_cov = np.cov(selected_scores.T)

            try:
                selected_inv_cov = np.linalg.inv(selected_cov)
            except np.linalg.LinAlgError:
                selected_cov = selected_cov + 1e-6 * np.eye(selected_cov.shape[0]) # add small value to diagonal to make it invertible
                selected_inv_cov = np.linalg.inv(selected_cov)

            # discard the covariance with the true answer
            selected_inv_cov = selected_inv_cov[:, :-1]
            selected_inv_cov = selected_inv_cov[:-1, :]

            # set diagonal to 0 (we don't count dependencies with itself)
            np.fill_diagonal(selected_inv_cov, 0)

            # record largest magnitude element 
            s = np.abs(selected_inv_cov).max()
            triple_to_sparsity[tuple(triple)] = s
            
            marginals = votes[:, triple].mean(axis=0)
            triple_to_marginal[tuple(triple)] = marginals

        sorted_sparsity = {k: v for k, v in sorted(triple_to_sparsity.items(), key=lambda x: x[1])}

        top_triple = list(sorted_sparsity.keys())[0]
        triple_names = [v for i, v in enumerate(current_verifiers) if i in top_triple]
        print(f"Top triple: {triple_names}, sparsity: {sorted_sparsity[top_triple]}", flush=True)
        if any(triple_to_marginal[top_triple] > 0.9) or any(triple_to_marginal[top_triple] < 0.1):
            print(f"WARNING: Some of the verifiers in the top triple have marginal probabilities that are too extreme: {triple_names}, {triple_to_marginal[top_triple]}", flush=True)

        top_triple = np.array(list(top_triple))
        votes = votes[:, top_triple]
        return votes, top_triple

    def get_nb_accs_and_cb(self, X, y):
        # binary_scores: n x n_verifiers
        # true_labels: n 
        assert X.ndim == 2

        indices_0 = np.where(y == 0)[0]
        indices_1 = np.where(y == 1)[0]

        if self.drop_imbalanced_verifiers is not None :
            marginals = X.mean(axis=0)
            if self.drop_imbalanced_verifiers == 'all':
                balanced_idxs = np.where((marginals > 0.1) & (marginals < 0.9))[0]
            elif self.drop_imbalanced_verifiers == 'small':
                balanced_idxs = np.where(marginals > 0.5)[0]
            elif self.drop_imbalanced_verifiers == 'large':
                balanced_idxs = np.where(marginals < 0.5)[0]
            else:
                raise ValueError(f"Unknown value for drop_imbalanced_verifiers: {self.drop_imbalanced_verifiers}")

            balanced_names = [v for i, v in enumerate(self.verifier_names) if i in balanced_idxs]
            print(f"Only keeping verifiers {balanced_names}", flush=True)
            X = X[:, balanced_idxs]
            self.verifier_idxs = balanced_idxs

        if self.use_deps == "drop":
            X, remaining_idxs = self._drop_deps(X, y, current_verifiers=balanced_names if self.drop_imbalanced_verifiers is not None else self.verifier_names)
            self.verifier_idxs = self.verifier_idxs[remaining_idxs]
            print(f"verifier_idxs: {self.verifier_idxs}", flush=True)

        num_verifiers = X.shape[1]
        tnr = np.zeros(num_verifiers)
        tpr = np.zeros(num_verifiers)

        for i in range(num_verifiers):
            verifier_scores = X[:, i]
            tnr[i] = accuracy_score(y[indices_0], verifier_scores[indices_0])
            tpr[i] = accuracy_score(y[indices_1], verifier_scores[indices_1])
    
        tpr = np.clip(tpr, self.clip_min, self.clip_max)
        tnr = np.clip(tnr, self.clip_min, self.clip_max)

        cb = y.mean()

        return tpr, tnr, cb

    def fit(self, X, y, model_idx=None):
        """Calculate tpr and tnr for each verifier."""
        X = self.preprocess(X)
        self.verifier_idxs = np.arange(X.shape[1])
        self.tpr, self.tnr, self.cb = self.get_nb_accs_and_cb(X, y)

        print(f"NB TPR: {self.tpr}, NB TNR: {self.tnr}", flush=True)

    def calculate_metrics(self, X, y):
        X = self.preprocess(X)
        X = X[:, self.verifier_idxs]
        tpr = self.tpr
        tnr = self.tnr

        fpr = 1 - self.tnr
        fnr = 1 - self.tpr

        # Compute log-likelihoods
        log_likelihood_y1 = np.sum(
            np.log(X * tpr + (1 - X) * fnr), axis=-1
        )  # (problems x samples)

        log_likelihood_y0 = np.sum(
            np.log(X * fpr + (1 - X) * tnr), axis=-1
        )  # (problems x samples)

        # Compute log posteriors (log-space multiplication turns into addition)
        log_posterior_y1 = log_likelihood_y1 + np.log(self.cb)
        log_posterior_y0 = log_likelihood_y0 + np.log(1 - self.cb)

        # Use logsumexp for stability: log(exp(a) + exp(b)) = logsumexp([a, b])
        log_prob_y1_given_features = log_posterior_y1 - logsumexp([log_posterior_y1, log_posterior_y0], axis=0)

        # Convert back to probability space
        prob_y1_given_features = np.exp(log_prob_y1_given_features)

        probs = np.array([1 - prob_y1_given_features, prob_y1_given_features]).T

        outputs1 = calculate_sample_metrics(probs, y)
        outputs2 = calculate_top1_metrics(probs, y)
        outputs3 = calculate_prediction_metrics(probs, y)
        outputs = {**outputs1, **outputs2, **outputs3}

        outputs["model_params"] = [self.tpr, self.tnr, fpr, fnr]
        outputs["verifier_subset"] = [v for i, v in enumerate(self.verifier_names) if i in self.verifier_idxs]
    
        return outputs


class MajorityVote(Model):
    """
    Majority@k:
      - For each problem, collect all answers and correctness.
      - Group identical answers.
      - Take the top-k most frequent answers.
      - If any of those top-k is correct by majority (>50%), 
        the problem is counted correct.
    Returns either mean accuracy or per-problem array.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set additional attributes dynamically

    def fit(self, X, y, model_idx=None):
        """No training required for majority vote."""
        if model_idx is not None:
            self.is_trained[model_idx] = True
        else:
            self.is_trained = True


    def calculate_majority_at_k(self, all_answers, y, return_mean=False, return_idx=True): 
        """
        
        Args:
            - all_answers: list of lists of answers, Each list contains the answers for a single problem.
            - y: np.array of shape (num_problems, num_samples)
            - return_mean: if True, return the mean accuracy
            - return_idx: if True, return the index with the top-k answer
        return:
            - correct_problems: number of problems that are correct
            - all_problems: array of 0/1 indicating if the problem is correct
            - top_k_idx: index of the top-k answer
        """
        k = self.k
        correct_problems = 0
        total_problems = 0

        num_problems = len(all_answers)
        all_problems = np.zeros(num_problems)

        all_topk_idx = np.zeros((num_problems, k))*np.nan
        # for each problem, count the number of times each answer appears
        # and check if the answer is correct by majority
        for problem_idx in range(num_problems):
            answers = all_answers[problem_idx]
            answers_correct = y[problem_idx]
            answer_counts, answer_correctness, answer_idx = {}, {}, {}
            # count the number of times each answer appears
            # answer_counts: {answer: count}
            # answer_correctness: {answer: [correctness]}
            # answer_idx: {answer: [index]}
            for idx, (ans, is_correct) in enumerate(zip(answers, answers_correct)):
                if ans == 'NO_ANSWER':
                    continue
                if ans not in answer_counts:
                    answer_counts[ans] = 0
                    answer_correctness[ans] = []
                    answer_idx[ans] = []
                answer_counts[ans] += 1
                answer_correctness[ans].append(is_correct)
                answer_idx[ans].append(idx)

            if answer_counts:
                total_problems += 1
                # Dictionary of top-k answers and their counts
                top_k_answers = sorted(
                    answer_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:k]

            # top_k_answers: list of tuples (answer, count)
            # answer_idx: dictionary of answer to list of indices
            # top_k_idx: list of k elements with indices of the top k answers
            top_k_idx = [answer_idx[ans] for ans, _ in top_k_answers]

            # Pick the first sample as the sample
            sample_idx = 0
            # topk_positive: number of queries where the top k answer is positive(correct)
            # prediction_accuracy: accuracy of the prediction (selected sample) for each query
            if self.majority_select == "majority":
                # List of lists of indices
                for top_id, (ans, _) in enumerate(top_k_answers):
                    # If this answer is correct by majority
                    if sum(answer_correctness[ans]) > len(answer_correctness[ans]) / 2:
                        correct_problems += 1
                        all_problems[problem_idx] = 1
                        # Pick any of the top k answers as the sample
                        all_topk_idx[problem_idx] = top_k_idx[top_id][sample_idx]
                        break 
            elif self.majority_select == "one_sample":
                for top_id, (ans, _) in enumerate(top_k_answers):
                    correct_sample = answer_correctness[ans][sample_idx]
                    if correct_sample:
                        correct_problems += 1
                        all_problems[problem_idx] = 1
                        all_topk_idx[problem_idx] = top_k_idx[top_id][sample_idx]
                        break

        if return_mean:
            output1 = correct_problems / total_problems
        else:
            output1 = all_problems

        if return_idx:
            output2 = all_topk_idx
        else:
            output2 = None

        return output1, output2
    
    def calculate_metrics(self, X, y, return_mean=True, return_idx=True):
        # Check 
        is_one_problem =  not is_list_of_lists(X)
        if is_one_problem:
            X = [X]
            y = [y]

        all_acc, topk_idx = self.calculate_majority_at_k(X, y, return_mean=return_mean, return_idx=return_idx)
        top1_correct = all_acc

        # Accuracy is proportion of problems that are correct
        if is_one_problem and not return_mean:
            top1_correct = top1_correct[0]
            topk_idx = topk_idx[0]
            all_acc = all_acc[0]

        # accuracy of the top k answer
        prediction_accuracy = all_acc # TODO: check
        model_params = None 
        verifier_subset = None

        if return_idx:
            # ANy of the responses have a true response
            y_true = np.asarray([np.any(y[problem_idx]) for problem_idx in range(len(y))]).astype(int)
            # ANy of the responses have a false response
            top1_label = np.asarray(
                [y[problem_idx][int(idx)] if (idx is not None and not np.isnan(idx)) else np.nan
                for problem_idx, idx in enumerate(topk_idx) ])
            top1_tp = int(top1_label == 1 and y_true == 1)  # picked something and the query had a true answer
            top1_fp = int(top1_label == 1 and y_true == 0)  # picked something in a query with no positives
            top1_tn = int(top1_label == 0 and y_true == 0)  # did NOT pick anything, and none existed
            top1_fn = int(top1_label == 0 and y_true == 1)  # missed something in a query with a true answer

        return {
            "top1_positive": top1_correct,
            "top1_idx": topk_idx,
            "sample_accuracy": all_acc,
            "prediction_accuracy": prediction_accuracy,
            "model_params": model_params,
            "verifier_subset": verifier_subset,
            "top1_tp": top1_tp,
            "top1_tn": top1_tn,
            "top1_fp": top1_fp,
            "top1_fn": top1_fn,
        }


class NaiveEnsemble(Model):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set additional attributes dynamically

    def fit(self, X, y):
        pass

    def calculate_metrics(self, X, y):
        assert X.ndim == 2 # n_samples x n_verifiers
        
        y_tmp = np.mean(X, axis=1)  # n_samples

        probs = np.array([1 - y_tmp, y_tmp]).T

        outputs1 = calculate_sample_metrics(probs, y)
        outputs2 = calculate_top1_metrics(probs, y)
        outputs3 = calculate_prediction_metrics(probs, y)
        outputs = {**outputs1, **outputs2, **outputs3}

        outputs["model_params"] = None
        outputs["verifier_subset"] = None
        return outputs


class FirstSample:
    def __init__(self, verifier_names=None, **kwargs):
        self.verifier_names = verifier_names
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        self.is_test = False
    
    def fit(self, X, y, **kwargs):
        pass  # No fitting needed for first sample selection
    
    def calculate_metrics(self, X, y, **kwargs):
        """
        Selects the first sample's prediction as the final decision.
        Note: This expects X to be the extracted answers (binary), not verifier scores.
        """
        # X should be the extracted answers, but we're getting verifier scores
        # We need to work with what we have - just use the ground truth labels
        
        # Always select first sample (index 0)
        top1_idx = 0
        first_sample_correct = y[0]  # Is the first sample correct?
        
        top1_positive = float(first_sample_correct)
        sample_accuracy = float(np.mean(y))
        prediction_accuracy = top1_positive
        
        has_positive = int(np.any(y == 1))
        
        top1_tp = int(first_sample_correct == 1 and has_positive == 1)
        top1_fp = int(first_sample_correct == 1 and has_positive == 0)
        top1_tn = int(first_sample_correct == 0 and has_positive == 0) 
        top1_fn = int(first_sample_correct == 0 and has_positive == 1)
        
        sample_tp = int(np.sum(y == 1))
        sample_tn = int(np.sum(y == 0))
        sample_fp = 0
        sample_fn = 0
        
        return {
            "top1_positive": top1_positive,
            "top1_idx": top1_idx,
            "sample_accuracy": sample_accuracy,
            "prediction_accuracy": prediction_accuracy,
            "model_params": None,
            "verifier_subset": None,
            "top1_tp": top1_tp,
            "top1_fp": top1_fp,
            "top1_tn": top1_tn,
            "top1_fn": top1_fn,
            "sample_tp": sample_tp,
            "sample_tn": sample_tn,
            "sample_fp": sample_fp,
            "sample_fn": sample_fn,
            "num_samples": len(y),
            "has_positive": has_positive,
        }


class Coverage(Model):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
    
    def fit(self, X, y, model_idx=None):
        pass  # No fitting needed for first sample selection
    
    def calculate_metrics(self, X, y):
        """
        Selects if any of the labels are correct
        y is a 1D array with shape (num_samples,)
        """
        top1_correct = np.any(y == 1).mean()
        model_params = None
        verifier_subset = None

        return {
            "top1_positive": top1_correct,
            "top1_idx": None,
            "sample_accuracy": top1_correct,
            "prediction_accuracy": top1_correct,
            "model_params": model_params,
            "verifier_subset": verifier_subset
        }
    

def is_list_of_lists(variable):
    return isinstance(variable, list) and all(isinstance(item, list) for item in variable)


MODEL_TYPES = {
    "logistic_regression": LogisticRegression2,
    "majority_vote": MajorityVote,
    "weak_supervision": WeakSupervised,
    "naive_bayes": NaiveBayes,
    "naive_ensemble": NaiveEnsemble,
    "first_sample": FirstSample,
    "coverage": Coverage,
    "unsupervised": Unsupervised,
    "unsupervised2": Unsupervised2,
    "first_sample": FirstSample,
}


def calculate_top1_correct(pred_, y):
    """Get the sample with the highest probability for class 1 and check if it is correct.
    
    pred_ is a 2D array with shape (num_samples, 2)
    y is a 1D array with shape (num_samples,)
    """
    assert pred_.shape[1] == 2  # Two classes (0 or 1)
    top1_idx = np.argmax(pred_[:, 1])  # Get index of max class 1 probability
    return int(y[top1_idx] == 1), top1_idx  # Return 1 if correct, 0 otherwise


def multivariate_normal_pdf_fast(X, mean, cov):
    """Fast implementation of multivariate normal PDF using Cholesky decomposition
    Use as replacement for scipy.stats.multivariate_normal.pdf
    """
    n_dim = len(mean)
    dev = X - mean
    
    # Compute Cholesky decomposition once
    try:
        chol = np.linalg.cholesky(cov)
        # Solve triangular system
        solved = solve_triangular(chol, dev.T, lower=True)
        logpdf = -0.5 * np.sum(solved**2, axis=0) - n_dim/2 * np.log(2*np.pi) - np.sum(np.log(np.diag(chol)))
        return np.exp(logpdf)
    except np.linalg.LinAlgError:
        # Fall back to standard method if Cholesky fails (non-positive definite matrix)
        return multivariate_normal.pdf(X, mean=mean, cov=cov)