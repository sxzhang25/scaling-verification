import os
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist 
from collections import Counter 
from weaver.constants import (
    DATASET_TO_REWARD_MODELS,
    DATASET_TO_LM_JUDGES,
    VERIFIER_NAME_MAP,
    DATASET_TO_HF,
    VERIFIER_DESCRIPTIONS,
)
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import QuantileTransformer
from scipy.stats.mstats import winsorize
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
import itertools


def from_cluster_list_to_dict(labels):
    """
    From a list of labels, create a dictionary of 
    cluster_id: [indices]
    Args:
        labels: The labels to create clusters from.
    Returns:
        A dictionary where the keys are cluster indices and the values are lists of problem indices.
    """
    clusters = defaultdict(list)
    for i, group in enumerate(labels):
        clusters[group].append(i)
    return clusters


def from_cluster_dict_to_list(clusters):
    """
    From a dictionary of cluster_id: [indices], 
    create a list of cluster_ids[indices]
    """
    num_elements = sum(len(v) for v in clusters.values())
    # get all the elements 
    reverse_map = np.zeros(num_elements)
    for cluster_id, cluster_elems in clusters.items():
        # idx is the cluster_id
        # cluster_elems is the elements in the cluster
        if len(cluster_elems) == 0:
            continue
        for cluster_elem in cluster_elems:
            reverse_map[cluster_elem] = cluster_id   
    return reverse_map.astype(int)


class ClusteringDataset:
    def __init__(self, cluster_type, n_clusters, **kwargs):
        self.cluster_type = cluster_type
        self.n_clusters = n_clusters
        self.preserve_ties = kwargs.get("preserve_ties", True)  # Default to True
        self.uniform_with_ties = kwargs.get("uniform_with_ties", False)
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        self.seed = kwargs.get('seed', 42)
        self.embedding_model = kwargs.get('embedding_model', "nomic-ai/modernbert-embed-base")
        self.cluster_on_all = kwargs.get('cluster_on_all', False)

        # Initialize parameters we need to keep track of for test set:
        self.bin_ids = None
        self.bin_edges = None
        self.cluster_model = None
        self.train_clusters = None
        self.test_clusters = None
        self.rng = np.random.default_rng(self.seed)

    def compute_clusters(self, data, mode="train") -> dict:
        """
        Compute clusters of train problems based on the specified clustering method.

        Args:
            data: The dataset containing the train problems.
            n_clusters: The number of clusters to create.
            cluster_type: The type of clustering to perform. Options are:
                - "random": Randomly assign train problems to clusters.
                - "by_difficulty": Cluster by difficulty.
                - "by_binned_difficulty": Cluster by binned difficulty.
                - "bert_query": Cluster using BERT embeddings.
                - "unique_extracted_answers": Cluster by number of unique extracted answers.
        Returns:    
            A dictionary where the keys are cluster indices and the values are lists of train problem indices.
        """
        """
        We are going to compute clusters on the train and test problems.
        """
        # Get all the data:
        # if train and test set are different
        if mode == "train":
            n_problems = len(data.train_data[0])
            y_data = data.train_data[1]
            input_data = data.train_problems
            data_indices = data.train_idx
        elif mode == "test":
            n_problems = len(data.test_data[0])
            y_data = data.test_data[1]
            input_data = data.test_problems
            data_indices = data.test_idx
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # For some types of clustering we need to fit on train and test data together:
        # i.e. hierarchical, dbscan, spectral, gmm
        fitting_on_train_and_test = False
        set_train_and_test_equal = np.array_equal(data.train_data[0], data.test_data[0])
        if self.cluster_type in ["hierarchical", "dbscan", "spectral"] or self.cluster_on_all:
            if not(set_train_and_test_equal):                # not equal, so we fit together
                fitting_on_train_and_test = True
                input_data = np.concatenate([data.train_problems, data.test_problems])
                y_data = np.concatenate([data.train_data[1], data.test_data[1]])
                n_problems = len(data.train_problems) + len(data.test_problems)
                data_indices = np.concatenate([data.train_idx, data.test_idx])

        if self.cluster_type == "random":
            # Does not use labels
            print(f"Randomly assigning {mode} problems to {self.n_clusters} clusters.", flush=True)
            shuffled = self.rng.permutation(np.arange(n_problems))

            # Calculate base chunk size and how many chunks need an extra element
            avg_chunk_size = n_problems // self.n_clusters
            remainder = n_problems % self.n_clusters
            clusters = {}
            start = 0
            cluster_ids = np.zeros(n_problems, dtype=int)
            for i in range(self.n_clusters):
                end = start + avg_chunk_size + (1 if i < remainder else 0)
                clusters[i] = shuffled[start:end]
                cluster_ids[shuffled[start:end]] = i
                start = end
        elif self.cluster_type == "by_difficulty":
            # TODO: update to not use labels on test set!
            if mode == "test":
                warnings.warn("Using ground truth labels for clustering test set.")
            # these clusters are balanced, but the bin ranges are not consistent:
            print(f"Clustering {mode} problems by difficulty into {self.n_clusters} clusters.", flush=True)
            problem_difficulty = y_data.mean(axis=1)
            sorted_indices = np.argsort(problem_difficulty, kind="stable")
            split_indices = np.array_split(sorted_indices, self.n_clusters)
            clusters = {i: np.array(group) for i, group in enumerate(split_indices)}
            cluster_ids = from_cluster_dict_to_list(clusters)
            print(f"Clusters {mode}: {clusters}", flush=True)
        elif self.cluster_type == "by_binned_difficulty":
            print(f"Clustering {mode} problems by binned difficulty into {self.n_clusters} clusters.", flush=True)
            # TODO: update to not use labels on test set!
            # clusters with uneven sizes but more consistent difficulty ranges.
            problem_difficulty = y_data.mean(axis=1)
            if mode == "train":
                # Step 1: Define bin edges for n_clusters histogram-style bins
                self.bin_edges = np.linspace(problem_difficulty.min(), problem_difficulty.max(), self.n_clusters)
                # Step 2: Assign each value to a bin (1 to n_clusters)
                bin_ids = np.digitize(problem_difficulty, bins=self.bin_edges, right=False)
                # Step 3: Clip any value that falls into bin n_clusters + 1 (edge case)
                bin_ids = np.clip(bin_ids, 1, self.n_clusters)
                self.bin_ids = bin_ids
            elif mode == "test":
                if self.bin_edges is None:
                    raise ValueError("You must compute clusters on training set before test.")
                warnings.warn("Using ground truth labels for clustering test set.")
                bin_ids = np.digitize(problem_difficulty, bins=self.bin_edges[:-1], right=False)
                bin_ids = np.clip(bin_ids, 1, self.n_clusters)

            # For test mode, use the same bin ids as the train set:
            # Step 4: Build clusters dict (0-based indexing for cluster keys)
            clusters = {i: np.where(bin_ids == i +1 )[0] for i in range(self.n_clusters)}
            cluster_ids = from_cluster_dict_to_list(clusters)
            
        elif self.cluster_type == "bert_query":
            print(f"Clustering train problems using {self.embedding_model} embeddings into {self.n_clusters} clusters.", flush=True)
            model = SentenceTransformer(self.embedding_model)
            encoded_data = model.encode(input_data)

            clusters = defaultdict(list)
            if mode == "train":
                self.cluster_model = KMeans(n_clusters=self.n_clusters,
                                random_state=self.seed,
                                n_init="auto").fit(encoded_data)

                for i, group in enumerate(self.cluster_model.labels_):
                    clusters[group].append(i)
                cluster_ids = self.cluster_model.labels_
            elif mode == "test":
                if self.cluster_model is None:
                    raise ValueError("KMeans model not trained. Run with mode='train' first.")
                cluster_ids = self.cluster_model.predict(encoded_data)
                for i, group in enumerate(cluster_ids):
                    clusters[group].append(i)
        elif self.cluster_type == "unique_extracted_answers":
            print(f"Clustering {mode} problems by unique extracted answers into {self.n_clusters} clusters.", flush=True)
            
            # Get extracted answers for this mode
            if mode == "train":
                answers = data.train_answers
            elif mode == "test":
                answers = data.test_answers
            
            if fitting_on_train_and_test:
                answers = np.concatenate([data.train_answers, data.test_answers])

            # Count unique answers for each problem
            unique_answer_counts = []
            for extracted in answers:
                unique_answers = len(set(x for x in extracted if x is not None))
                unique_answer_counts.append(unique_answers)
            
            unique_answer_counts = np.array(unique_answer_counts)
            
            if self.uniform_with_ties:
                # Group questions by their unique answer count
                unique_counts = np.unique(unique_answer_counts)[::-1]  # Sort in descending order
                count_to_questions = {count: np.where(unique_answer_counts == count)[0] for count in unique_counts}
                
                # Calculate target bucket size
                target_bucket_size = len(unique_answer_counts) / self.n_clusters
                
                # Assign buckets while trying to maintain uniform size
                buckets = np.zeros(len(unique_answer_counts), dtype=int)
                current_bucket = 0
                current_bucket_size = 0
                
                for count in unique_counts:
                    questions = count_to_questions[count]
                    questions_in_group = len(questions)
                    
                    # If adding this group would make the current bucket too large,
                    # decide whether to put it in the current bucket or start a new one
                    if current_bucket_size + questions_in_group > target_bucket_size * 1.5 and current_bucket < self.n_clusters - 1:
                        # If current bucket is very small, add this group to it
                        if current_bucket_size < target_bucket_size * 0.5:
                            buckets[questions] = current_bucket
                            current_bucket_size += questions_in_group
                        else:
                            # Start a new bucket
                            current_bucket += 1
                            buckets[questions] = current_bucket
                            current_bucket_size = questions_in_group
                    else:
                        # Add to current bucket
                        buckets[questions] = current_bucket
                        current_bucket_size += questions_in_group
                        
                        # Start new bucket if current one is full enough
                        if current_bucket_size >= target_bucket_size and current_bucket < self.n_clusters - 1:
                            current_bucket += 1
                            current_bucket_size = 0
                
            elif self.preserve_ties:
                # Get sorted unique values of answer counts
                distinct_counts = np.sort(np.unique(unique_answer_counts))[::-1]
                
                # Calculate bucket boundaries based on number of distinct values
                bucket_boundaries = np.array_split(distinct_counts, self.n_clusters)
                
                # Assign buckets based on which boundary group contains the count
                buckets = np.zeros(len(unique_answer_counts), dtype=int)
                for i, count in enumerate(unique_answer_counts):
                    for bucket, boundary_group in enumerate(bucket_boundaries):
                        if count in boundary_group:
                            buckets[i] = bucket
                            break
            else:
                # Original uniform bucketing without preserving ties
                # Sort indices by unique answer counts in descending order (more unique answers = harder = lower bucket)
                sorted_indices = np.argsort(-unique_answer_counts, kind="stable")
                
                # Calculate size of each bucket
                bucket_size = len(unique_answer_counts) // self.n_clusters
                remaining = len(unique_answer_counts) % self.n_clusters
                
                # Assign buckets
                buckets = np.zeros(len(unique_answer_counts), dtype=int)
                current_pos = 0
                
                # Distribute questions into buckets of uniform size
                # If there are remaining questions, add one extra to the first 'remaining' buckets
                for bucket in range(self.n_clusters):
                    bucket_end = current_pos + bucket_size + (1 if bucket < remaining else 0)
                    buckets[sorted_indices[current_pos:bucket_end]] = bucket
                    current_pos = bucket_end
            
            # Create clusters dictionary
            clusters = {i: np.where(buckets == i)[0] for i in range(self.n_clusters)}
            cluster_ids = from_cluster_dict_to_list(clusters)
            # Print statistics
            print("\nBucket distribution:")
            for i in range(self.n_clusters):
                count = len(clusters[i])
                min_answers = min(unique_answer_counts[buckets == i]) if count > 0 else 0
                max_answers = max(unique_answer_counts[buckets == i]) if count > 0 else 0
                print(f"Bucket {i}: {count} questions ({len(buckets):.1f})")
                print(f"  Unique answer range: {min_answers} - {max_answers}")
                
            if self.preserve_ties:
                print("\nNote: Preserving ties may result in uneven bucket sizes")
            elif self.uniform_with_ties:
                print("\nNote: Attempting to maintain uniform bucket sizes while preserving ties where possible")

        elif self.cluster_type == "json":
            import json
            json_path = self.embedding_model  # Reuse embedding_model parameter as json_path
            print(f"Loading cluster assignments from JSON file: {json_path}", flush=True)
            try:
                with open(json_path, 'r') as f:
                    cluster_mapping = json.load(f)
                # Convert string keys to integers if needed
                cluster_mapping = {int(k): int(v) for k, v in cluster_mapping.items()}
                
                # Create a mapping from original indices to current indices
                # data_indices contains the current indices in the dataset
                # We need to find where each original index from the JSON maps to in data_indices
                index_map = {orig_idx: curr_idx for curr_idx, orig_idx in enumerate(data_indices)}
                
                # Convert to format expected by rest of code
                clusters = {}
                for prob_idx, cluster_idx in cluster_mapping.items():
                    if cluster_idx not in clusters:
                        clusters[cluster_idx] = []
                    # Only add if the original index exists in our current dataset
                    if prob_idx in index_map:
                        # Map the original index to the current index
                        mapped_idx = index_map[prob_idx]
                        clusters[cluster_idx].append(mapped_idx)
                
                # Convert lists to numpy arrays
                clusters = {k: np.array(v) for k, v in clusters.items()}
                print(f"Loaded {len(cluster_mapping)} problem assignments across {len(clusters)} clusters", flush=True)
                # Store cluster mapping for future use
                self.cluster_mapping = cluster_mapping
                cluster_ids = np.array([cluster_mapping[i] for i in data_indices])
            except Exception as e:
                raise ValueError(f"Failed to load or parse cluster JSON file: {e}")
            
            
        elif self.cluster_type == "hierarchical":
            print(f"Clustering all problems using hierarchical clustering into {self.n_clusters} clusters.", flush=True)
            from sklearn.cluster import AgglomerativeClustering
            
            # Get embeddings
            model = SentenceTransformer(self.embedding_model)
            X_train_repr = model.encode(input_data)
            
            # Apply hierarchical clustering
            if mode == "train":
                self.cluster_model = AgglomerativeClustering(
                    n_clusters=self.n_clusters, 
                    linkage='ward'  # Options: 'ward', 'complete', 'average', 'single'
                )

                hierarchical = self.cluster_model.fit(X_train_repr)


                # Create clusters
                clusters = defaultdict(list)
                for i, group in enumerate(hierarchical.labels_):
                    clusters[group].append(i)
                cluster_ids = hierarchical.labels_
            else:
                pass
        elif self.cluster_type == "dbscan":
            print(f"Clustering all problems using density-based clustering (DBSCAN).", flush=True)
            from sklearn.cluster import DBSCAN
            
            # Get embeddings
            if mode == "train":
                model = SentenceTransformer(self.embedding_model)
                X_train_repr = model.encode(input_data)
                
                # Apply DBSCAN
                # These parameters may need tuning based on your data
                eps = 0.5  # Distance threshold
                min_samples = 5  # Min points to form dense region
                
                self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
                self.cluster_model.fit(X_train_repr)
                labels = self.cluster_model.labels_
                # Handle noise points (label -1) by assigning them to their own cluster
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"DBSCAN found {n_clusters_found} clusters and {np.sum(labels == -1)} noise points", flush=True)
                
                # Create clusters
                clusters = defaultdict(list)
                cluster_id = 0
                for i, group in enumerate(labels):
                    if group == -1:  # Noise point gets its own cluster
                        clusters[n_clusters_found + cluster_id].append(i)
                        cluster_id += 1
                    else:
                        clusters[group].append(i)
                cluster_ids = labels
            else:
                pass
        elif self.cluster_type == "spectral":
            print(f"Clustering all problems using spectral clustering into {self.n_clusters} clusters.", flush=True)
            from sklearn.cluster import SpectralClustering
            
            if mode == "train":
                # Get embeddings
                model = SentenceTransformer(self.embedding_model)
                X_train_repr = model.encode(input_data)
                
                # Apply spectral clustering
                spectral = SpectralClustering(
                    n_clusters=self.n_clusters,
                    random_state=0,
                    affinity='nearest_neighbors'  # Options: 'nearest_neighbors', 'rbf'
                ).fit(X_train_repr)
                
                # Create clusters
                clusters = defaultdict(list)
                for i, group in enumerate(spectral.labels_):
                    clusters[group].append(i)
                cluster_ids = spectral.labels_
            else:
                pass
        elif self.cluster_type == "gmm":
            print(f"Clustering all problems using Gaussian Mixture Model into {self.n_clusters} clusters.", flush=True)
            from sklearn.mixture import GaussianMixture
            
            # Get embeddings
            model = SentenceTransformer(self.embedding_model)
            X_train_repr = model.encode(input_data)
            
            # Apply GMM
            if mode == "train":
                self.cluster_model = GaussianMixture(
                    n_components=self.n_clusters,
                    covariance_type='full',  # Options: 'full', 'tied', 'diag', 'spherical'
                    random_state=self.seed,
                    n_init=10
                )
                self.cluster_model.fit(X_train_repr)  
       
            # Assign each point to its most likely cluster
            labels = self.cluster_model.predict(X_train_repr)
            
            # Create clusters
            clusters = defaultdict(list)
            for i, group in enumerate(labels):
                clusters[group].append(i)
            cluster_ids = from_cluster_dict_to_list(clusters)
        else:
            # avg_i Emb([query, response_i])
            # - two problems for which the same verifier is the best belong to the same cluster 
            # i.e. grouping based on verifier accuracy 
            # - (bad when taken to extreme): clustering by exact profile of verifier outputs (all 30-something)
            # but we can cluster based on subset of profile (top 5)
            raise NotImplementedError(f"Cluster type {self.cluster_type} invalid")


        # ------------------------------------------------------------
        # Now we can set the clusters:
        if mode == "train":
            assert len(cluster_ids) == len(data_indices)
            self.train_cluster_idxs = cluster_ids
            self.train_clusters = from_cluster_list_to_dict(cluster_ids)
            if set_train_and_test_equal:
                self.test_cluster_idxs = cluster_ids
                self.test_clusters = from_cluster_list_to_dict(cluster_ids) 

            if fitting_on_train_and_test:
                # Test clusters 
                num_train_problems = len(data.train_data[0])
                train_cluster_idxs = cluster_ids[:num_train_problems]
                test_cluster_idxs = cluster_ids[num_train_problems:]

                self.train_cluster_idxs = train_cluster_idxs
                self.test_cluster_idxs = test_cluster_idxs
                self.train_clusters = from_cluster_list_to_dict(train_cluster_idxs)   
                self.test_clusters = from_cluster_list_to_dict(test_cluster_idxs)

        elif mode == "test":
            if not(fitting_on_train_and_test or set_train_and_test_equal):
                # when test not set in the train mode
                self.test_clusters = clusters
                self.test_cluster_idxs = cluster_ids

        # also match to train_idx:


    def find_test_set_clusters(self, data):
        # Assign clusters to the test set:
        self.compute_clusters(data, mode="test")
        train_clusters = self.train_clusters
        test_clusters = self.test_clusters

        if self.cluster_type == "random":
            # cluster assignment is random, so we can just return the closest train idxs
            return data.closest_train_idxs

        num_train_problems = len(self.train_cluster_idxs)
        if num_train_problems == 0:
            return data.closest_train_idxs

        reverse_map_train = np.zeros(len(data.train_data[0])).astype(int)
        for idx, cluster_id in train_clusters.items():
            for c in cluster_id:
                reverse_map_train[c] = idx

        reverse_map_test = np.zeros(len(data.test_data[0])).astype(int)
        for idx, cluster_id in test_clusters.items():
            for c in cluster_id:
                reverse_map_test[c] = idx

        # Reorganize the closest train idxs based on the clusters:
        num_test_problems = len(data.test_data[0])
        ranked_train_idxs = data.closest_train_idxs

        for idx in range(num_test_problems):
            # get the cluster id for the test set problem
            test_cluster = reverse_map_test[idx]
            # get the cluster id for the closest train problem

            closest_train_idx = ranked_train_idxs[idx]
            # filter out indices not in the same cluster as the test set problem
            # closest_train_idx is an array of indices 
            train_cluster = reverse_map_train[closest_train_idx]

            # any cluster that is not the same should be removed
            # find the indices of the train clusters that are the same as the test cluster
            same_cluster_idxs = closest_train_idx[np.where(train_cluster== test_cluster)[0]]
            if len(same_cluster_idxs) > 0:
                ranked_train_idxs[idx, :len(same_cluster_idxs)] = same_cluster_idxs
                ranked_train_idxs[idx, len(same_cluster_idxs):] = same_cluster_idxs[0]
        return ranked_train_idxs

class Normalizer:
    def __init__(self, normalize_method, normalize_params, random_seed, verifier_name=None, global_min=False, global_max=False):
        """
        Normalize score to be in range [0,1]

        Args:
            normalize_method (str): Method to normalize the score.
            normalize_params (dict): Parameters for the normalization method.
            normalize_type (str): Type of normalization to perform.
            random_seed (int): Random seed for reproducibility.
            global_min (float): Global minimum value for normalization.
        """
        self.normalize_method = normalize_method
        self.normalize_params = normalize_params
        self.random_seed = random_seed
        self.global_min = global_min
        self.global_max = global_max
        self.verifier_name = verifier_name
        
    # TODO: use verifier_name information

    def normalize_minmax(self, X):
        # Check if X is unique
        if self.check_unique(X):
            return (X - self.global_min) / (self.global_max - self.global_min)
        else:
            x_min = np.nanmin(X)
            y_max = np.nanmax(X)
            return (X - x_min) / (y_max - x_min)

    def normalize_quantile(self, X):
            normalize_params = self.normalize_params
            normalize_params['n_quantiles'] = min(len(X), normalize_params.get('n_quantiles', 1000))
            transformer = QuantileTransformer(**normalize_params,
                                                random_state=self.random_seed)
            output =  transformer.fit_transform(X.reshape(-1,1)).reshape(X.shape)

            # here output may not be in range [0,1]
            if not self.check_range(output):
                # apply minmax normalization
                output = self.normalize_minmax(output)

            return output

    def normalize_winsorize(self, X):
        lower_quantile = self.normalize_params.get('lower_quantile', 0.05)
        upper_quantile = self.normalize_params.get('upper_quantile', 0.95)
        output = winsorize(X, limits=(lower_quantile, upper_quantile))

        # here output may not be in range [0,1]
        if not self.check_range(output):
            # apply minmax normalization
            output = self.normalize_minmax(output)
        return output

    def check_range(self, X):
        if np.nanmin(X) >= 0.0 and np.nanmax(X) <= 1.0: 
            return True
        else:
            return False

    def check_nan(self, X):
        if np.isnan(X).all():
            return True
        else:
            return False

    def check_unique(self, X):
        if len(np.unique(X)) == 1:
            return True
        else:
            return False

    def normalize_score(self, X):
        # if X is all nan, return X
        if self.check_nan(X):
            return X
        # if X in the range [0,1], return X
        if self.check_range(X):
            return X

        if self.normalize_method == "minmax":
            output = self.normalize_minmax(X)

        elif self.normalize_method == "quantile":
            output = self.normalize_quantile(X)

        elif self.normalize_method == "winsorize":
            output = self.normalize_winsorize(X)

        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")

        # if quantile is mapped to gaussian, then output is not in range [0,1]
        # if method is winsize, likewise:
        if (self.normalize_method == "quantile" and self.normalize_params.get("output_distribution", "uniform") == "normal") or self.normalize_method == "winsorize":
            output = self.normalize_minmax(output)

        # MAKE SURE OUTPUT IS IN RANGE [0,1]
        if not self.check_range(output):
            raise ValueError(f"Output is not in range [0,1]: {output}")
        return output


class VerificationDataset:
    def __init__(self, dataset_name, model_size, **kwargs):
        """
        Initializes the Dataset object.

        Args:
            dataset_name (str): Name of the dataset.
            model_size (str): Model size to load data for.
            **kwargs: Additional parameters for customization.
        """
        self.dataset_name = dataset_name
        self.model_size = model_size
        self.__dict__.update(kwargs)  # Set additional attributes dynamically
        self.rng = np.random.default_rng(self.random_seed)
        
        self.train_split_bins = kwargs.get('train_split_bins', 10)
        self.drop_imbalanced_verifiers = kwargs.get("drop_imbalanced_verifiers", None)
        self.use_deps = kwargs.get("use_deps", None)
        self.max_num_independent_verifiers = kwargs.get("max_num_independent_verifiers", 3)

        # Add verifier augmentation config with defaults
        self.verifier_augmentation = kwargs.get('verifier_augmentation', {
            'smoothing': False,
            'interpolation': False,
            'calibration': False,
            'smoothing_window': 3,
            'calibration_method': 'isotonic'  # 'isotonic' or 'sigmoid'
        })
        
        self.dataset_path = kwargs.get('dataset_path', None)
        
        if self.dataset_path:
            if dataset_name:
                print(f"Warning: Both dataset_path and dataset_name provided. Using dataset_path: {self.dataset_path}")
            self.dataset_mapping = self.dataset_path  # Set mapping to custom path
        else:
            self.dataset_mapping = DATASET_TO_HF[dataset_name][model_size]  # Use predefined mapping
            
        self.split_training_data()

    def _smooth_verifier_scores(self, X):
        """Apply temporal smoothing to verifier scores."""
        if not self.verifier_augmentation['smoothing']:
            return X
            
        window = self.verifier_augmentation['smoothing_window']
        smoothed = np.zeros_like(X)
        
        for i in range(X.shape[0]):  # for each problem
            for j in range(X.shape[2]):  # for each verifier
                scores = X[i, :, j]
                # Apply moving average smoothing
                smoothed[i, :, j] = np.convolve(scores, np.ones(window)/window, mode='same')
                
        return smoothed

    def _interpolate_missing_scores(self, X):
        """Interpolate missing verifier scores using other verifiers."""
        if not self.verifier_augmentation['interpolation']:
            return X
            
        interpolated = X.copy()
        
        for i in range(X.shape[0]):  # for each problem
            for j in range(X.shape[1]):  # for each sample
                missing_mask = np.isnan(X[i, j, :])
                if np.any(missing_mask) and not np.all(missing_mask):
                    # If there are missing values but not all are missing
                    present_scores = X[i, j, ~missing_mask]
                    present_verifiers = np.arange(X.shape[2])[~missing_mask]
                    missing_verifiers = np.arange(X.shape[2])[missing_mask]
                    
                    # Use linear interpolation based on other verifiers
                    for v in missing_verifiers:
                        # Find closest verifiers that have scores
                        distances = np.abs(present_verifiers - v)
                        closest = np.argsort(distances, kind="stable")[:2]  # use 2 closest verifiers
                        weights = 1 / (distances[closest] + 1e-6)
                        weights = weights / weights.sum()
                        interpolated[i, j, v] = np.sum(present_scores[closest] * weights)
                        
        return interpolated

    def _calibrate_verifier_scores(self, X):
        """Calibrate verifier scores to make them more comparable."""
        if not self.verifier_augmentation['calibration']:
            return X
            
        calibrated = X.copy()
        method = self.verifier_augmentation['calibration_method']
        
        for j in range(X.shape[2]):  # for each verifier
            scores = X[:, :, j]
            if method == 'isotonic':
                # Use isotonic regression for calibration
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds='clip')
                
                # Calibrate each sample's scores independently
                for i in range(scores.shape[0]):  # for each problem
                    sample_scores = scores[i, :]
                    valid_mask = ~np.isnan(sample_scores)
                    if np.any(valid_mask):
                        # Create target values (0 to 1) for calibration
                        target_values = np.linspace(0, 1, len(sample_scores[valid_mask]))
                        ir.fit(sample_scores[valid_mask], target_values)
                        # Transform and store calibrated scores
                        calibrated[i, :, j] = ir.transform(sample_scores)
            elif method == 'sigmoid':
                # Use sigmoid calibration
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression()
                # Flatten scores and fit calibration
                flat_scores = scores.flatten()
                valid_mask = ~np.isnan(flat_scores)
                if np.any(valid_mask):
                    # Create binary target values for calibration
                    target_values = (flat_scores[valid_mask] > np.median(flat_scores[valid_mask])).astype(int)
                    lr.fit(flat_scores[valid_mask].reshape(-1, 1), target_values)
                    # Transform and reshape back to original shape
                    calibrated[:, :, j] = lr.predict_proba(scores.reshape(-1, 1))[:, 1].reshape(scores.shape)
                    
        return calibrated

    def load_task_data(self):
        """
        Loads dataset for the specified task and model size from huggingface.
        If we have a dataset path, ignore dataset name and model size, load directly.
        Else, look up the dataset name x model size in our released dataset registry.
        Also gets a list containing the names of the verifiers used in this dataset.
        
        Returns:
            DataFrame containing dataset
            The key used to find the correct answer for this dataset ("answer_correct")
            A list of human readable verifier names used in this dataset
        """
        dataset_name = self.dataset_name
        model_size = self.model_size
        correct_key = "answer_correct"

        if self.dataset_path:
            print(f"Loading custom dataset from: {self.dataset_path}")
            # Users can pass local or remote dataset
            try:
                if os.path.exists(self.dataset_path):
                    df = pd.DataFrame(load_from_disk(self.dataset_path))
                else:
                    df = pd.DataFrame(load_dataset(self.dataset_path)["data"])

                # Dynamically discover the verifiers used in the custom dataset
                all_reward_models = [col for col in df.columns if col.endswith('_scores')]
                all_judges = [col for col in df.columns if col.endswith('_verdicts')]
            except:
                raise ValueError(f"Could not load dataset from {self.dataset_path}")
                
            self.dataset_mapping = self.dataset_path
        else:
            # Use dataset_name and model_size to look up the HF patah for this dataset
            print(f"Loading dataset: {dataset_name} with model size: {model_size}", flush=True)
            dataset = DATASET_TO_HF[self.dataset_name][self.model_size]
            
            # Look up verifiers used for this dataset using predetermined mapping
            all_reward_models = DATASET_TO_REWARD_MODELS.get(dataset, [])
            all_judges = DATASET_TO_LM_JUDGES.get(dataset, [])

            # Load the dataset from hub
            df = pd.DataFrame(load_dataset(dataset)["data"])
            self.dataset_mapping = dataset
        
        # Rename verifiers to human readable names
        all_reward_models = [VERIFIER_NAME_MAP.get(v, v) for v in all_reward_models]
        all_judges = [VERIFIER_NAME_MAP.get(v, v) for v in all_judges]
            
        # Rename verifiers in the actual dataframe using same map
        df = df.rename(columns=VERIFIER_NAME_MAP)

        # Check that there are no repeated columns:
        if not len(df.columns) == len(set(df.columns)):
            if not (dataset_name in ["MATH-500"]):
                raise ValueError(f"Repeated column names in {dataset_name} {model_size}")
            else:
                df = df.loc[:, ~df.columns.duplicated()]
                
        # Which verifiers to use:
        if self.verifier_cfg.get("verifier_type", "all") == "all":
            verifier_names = all_reward_models + all_judges
        elif self.verifier_cfg.verifier_type == "reward_models":
            verifier_names = all_reward_models
        elif self.verifier_cfg.verifier_type == "judges":
            verifier_names = all_judges
        elif self.verifier_cfg.verifier_type == "specific_subset":
            verifier_names = self.verifier_cfg.verifier_subset
            assert all(v in all_reward_models + all_judges for v in verifier_names), f"Unknown verifiers: {verifier_names} from list {all_reward_models + all_judges}"
        else:
            raise ValueError(f"Unknown verifier type: {self.verifier_cfg.verifier_type}")

        # Subselect by size as well
        verifier_sizes = [VERIFIER_DESCRIPTIONS[v]["num_parameters"] for v in verifier_names if v != "weaver_scores"]
        verifier_size = self.verifier_cfg.get("verifier_size", "all")
        if type(verifier_size) == str:
            if verifier_size == "all":
                pass
            elif verifier_size == "small": 
                # all verifiers with less than 8B parameters
                verifier_names = [v for v, size in zip(verifier_names, verifier_sizes) if not(size is None) and size <= 8.0]
            elif verifier_size == "medium":
                # all verifiers with more than 8B parameters and less than 32B parameters
                verifier_names = [v for v, size in zip(verifier_names, verifier_sizes) if not(size is None) and size > 8.0 or size < 32.0]
            elif verifier_size == "large":
            # all verifiers with more than 32B parameters and less than 70B parameters
                verifier_names = [v for v, size in zip(verifier_names, verifier_sizes) if not(size is None) and size >= 32.0]
            else:
                raise ValueError(f"Unknown verifier size: {self.verifier_cfg.verifier_size}")
        elif type(verifier_size) == int:
            verifier_names = [v for v, size in zip(verifier_names, verifier_sizes) if not(size is None) and size <= verifier_size]
        else:
            raise ValueError(f"Unknown verifier size: {self.verifier_cfg.verifier_size}")
        
        # Remove verifiers which we are not using:
        verifier_names = [v for v in verifier_names if "_step" not in v]

        if self.mv_as_verifier:
            print(f"Adding majority vote verifier to the dataset.", flush=True)
            verifier_names += ['mv_verifier']

        # Assert that all verifier names are unique:
        assert len(verifier_names) == len(set(verifier_names)), "Duplicate verifier names"

        return df, correct_key, verifier_names

    def preprocess(self, X, verifier_names):
        """Preprocesses verifier scores by normalizing them."""
        assert X.ndim == 3, "X must be a 3D array (num_problems, num_responses, num_verifiers)"
        num_problems, num_responses, num_verifiers = X.shape
        assert len(verifier_names) == num_verifiers, "Verifier name mismatch"

        X = X.astype(float)

        # Apply verifier augmentations
        X = self._smooth_verifier_scores(X)
        X = self._interpolate_missing_scores(X)
        X = self._calibrate_verifier_scores(X)

        # Mask problems and verifiers with only NaN values
        mask_problems = ~np.isnan(X).all(axis=(1, 2))
        mask_verifiers = ~np.isnan(X).all(axis=(0, 1))

        X = X[mask_problems][:, :, mask_verifiers]
        verifier_names = [v for v, keep in zip(verifier_names, mask_verifiers) if keep]

        # Initialize after filtering out problems and verifiers with only nan responses
        X_data = np.full(X.shape, np.nan) 
        for v_idx, verifier in enumerate(verifier_names):
            X_ = X[..., v_idx]  # num_problems x num_responses 
            
            # Compute global (population-level) min/max for fallback
            global_min, global_max = np.nanmin(X_), np.nanmax(X_)
            
            normalizer = Normalizer(normalize_method=self.normalize_method,
                                    normalize_params=self.normalize_params,
                                    random_seed=self.random_seed,
                                    verifier_name=verifier,
                                    global_min=global_min,
                                    global_max=global_max)
            
            if self.normalize_type == "per_problem":
                for i in range(num_problems):
                    X_data[i, :, v_idx] = normalizer.normalize_score(X_[i, :])
            elif self.normalize_type == "all_problems":
                X_data[..., v_idx] = normalizer.normalize_score(X_)
        
        # Replace nan with mean of the data:
        if self.nan_replacement == "mean":
            # mean along samples 
            replacement_value = np.nanmean(X_data, axis=1, keepdims=True)
        elif self.nan_replacement == 0:
            replacement_value = 0
        else:
            raise ValueError(f"Unknown nan_replacement: {self.nan_replacement}")
        X_data = np.nan_to_num(X_data, nan=replacement_value)

        return X_data, mask_problems, mask_verifiers


    def binarize_verifiers(self, clusters=None, split="train"):
        if split == "train":
            X_data, y = self.train_data
        elif split == "test":
            X_data, y = self.test_data

        num_problems, _, num_verifiers = X_data.shape

        assert self.reward_threshold is not None, "Reward threshold must be specified for binarization: float (e.g. 0.5), cb, cb_per_problem, cb_per_cluster"

        if type(self.reward_threshold) == float:
            print(f"Binarizing reward model outputs with threshold: {self.reward_threshold}", flush=True)
            X_data = (X_data >= self.reward_threshold).astype(int)
        elif self.reward_threshold == "cb":
            print(f"Binarizing reward model outputs using overall dataset difficulty (class balance)", flush=True)
            cb = y.mean() 
            threshold = []
            for i in range(num_verifiers):
                if np.array_equal(np.unique(X_data[:, :, i]), np.array([0, 1])):
                    # this is a judge, don't threshold
                    threshold.append(0.5)
                else:
                    sorted_verifier_scores = np.sort(X_data[:, :, i].flatten())
                    index = int(np.ceil((1-cb) * len(sorted_verifier_scores))) - 1
                    threshold.append(sorted_verifier_scores[index])
            threshold = np.array(threshold)
            X_data = (X_data >= threshold).astype(float)
        elif self.reward_threshold == "cb_per_problem":
            print("Binarizing reward model outputs using per problem difficulty (class balance)", flush=True)
            cb_per_problem = y.mean(axis=1) 

            # For judges, we set threshold to 0.5 (do nothing)
            verifier_is_binary = np.array([
                np.array_equal(np.unique(X_data[:, :, i]), [0, 1])
                for i in range(num_verifiers)
            ])

            threshold = np.zeros((num_problems, num_verifiers))
            threshold[:, verifier_is_binary] = 0.5

            # For non-binary verifiers, compute per-problem thresholds
            nonbinary_idx = np.where(~verifier_is_binary)[0]

            cb_indices = np.ceil((1 - cb_per_problem)[:, None] * X_data.shape[1]).astype(int) - 1
            cb_indices = np.clip(cb_indices, 0, X_data.shape[1] - 1)  # safety

            for i in nonbinary_idx:
                sorted_scores = np.sort(X_data[:, :, i], axis=1)  
                threshold[:, i] = sorted_scores[np.arange(num_problems), cb_indices[:, 0]]

            X_data = (X_data >= threshold[:, np.newaxis, :]).astype(float)
        elif self.reward_threshold == "cb_per_cluster":
            assert self.train_split == 1.0, "cb_per_cluster is only supported for train=test"
            assert clusters is not None, "cb_per_cluster requires clusters to be provided"
            print("Binarizing reward model outputs using per cluster difficulty (class balance)", flush=True)
            
            cb_per_cluster = {i: y[cluster_idxs].mean() for i, cluster_idxs in clusters.items()}
            threshold = np.zeros((len(clusters), num_verifiers))
            for i in range(num_verifiers):
                for j in clusters:
                    if np.array_equal(np.unique(X_data[:, :, i]), [0, 1]):
                        # this is a judge, don't threshold
                        continue
                    else:
                        sorted_verifier_scores = np.sort(X_data[clusters[j], :, i].flatten())
                        index = int(np.ceil((1-cb_per_cluster[j]) * len(sorted_verifier_scores))) - 1
                        threshold[j, i] = sorted_verifier_scores[index]

            # convert threshold per cluster into threshold per problem 
            threshold_per_n = np.zeros((num_problems, num_verifiers))
            for cluster_id, indices in clusters.items():
                threshold_per_n[indices] = threshold[cluster_id] 
            X_data = (X_data >= threshold_per_n[:, np.newaxis, :]).astype(float)

        # update X 
        if split == "train":
            self.train_data = (X_data, y)
        elif split == "test":
            self.test_data = (X_data, y)


    def add_mv_verifier(self, df):
        # create the majority verifier 'scorer'.
        if "mv_verifier" in df.columns:
            return df

        print(f"Creating MV as verifier.", flush=True)

        extracted_answers = np.array(df['extracted_answers'].values)
        mv_data = []
        for i, row in enumerate(extracted_answers):
            c = Counter(row)
            freqs = np.array(list(c.values()))
            freqs = freqs/freqs.sum() # normalize the frequencies
            if len(freqs) != 1:
                min_freq, max_freq = freqs.min(), freqs.max() 
                if min_freq == max_freq:
                    # if all extracted answers are equally likely, set score to 1 
                    freqs = np.ones_like(freqs) 
                else:
                    freqs = (freqs - min_freq)/(max_freq - min_freq) # scale them to be from 0 to 1 
            freqs = {ans: freqs[j]  for j, ans in enumerate(c.keys())} 
            mv_row = [freqs[ans] for ans in row ] # assign the normalized frequencies to the answers, as the mv scorer.
            mv_data.append(mv_row)

        df['mv_verifier'] = mv_data
        return df

    def setup(self):
        """Loads dataset, extracts verifier scores, and prepares training/testing splits."""
        # we want to load the task data
        df, correct_key, all_verifiers = self.load_task_data()
        y_data = np.stack(df[correct_key].values).astype(int)

        print(f"Number of problems: {len(y_data)} and samples: {len(y_data[0])}", flush=True)

        if "mv_verifier" in all_verifiers:
            df = self.add_mv_verifier(df) # add mv column to dataframe

        # Build X_data as (num_problems, num_responses, num_verifiers)
        # which has all the verifier scores for each response
        verifier_matrices = []
        verifier_names = []
        # data is num_problems x num_responses x num_verifiers
        for verifier in all_verifiers:
            raw_scores = df[verifier].values  # list of arrays
            raw_scores = np.stack(raw_scores, axis=0).squeeze()
            verifier_names.append(verifier)
            verifier_matrices.append(raw_scores)
        X_data = np.stack(verifier_matrices, axis=-1)  # shape: (num_problems, num_responses, num_verifiers)

        if self.dataset_name == "GPQA-Diamond":
            # use only a subset of the verifiers
            diamond_queries = (df["type"]=="diamond").values
            X_data = X_data[diamond_queries]
            y_data = y_data[diamond_queries]
            
            print(f"Filtering GPQA-Diamond problems. Number of problems: {len(y_data)} and samples: {len(y_data[0])}", flush=True)
        
        # Add extracted_answers to the dataset for majority vote
        try:
            answers = df["extracted_answers"].values
        except:
            answers = df["samples"].values

        problems = df["instruction"].values

        return X_data, y_data, verifier_names, answers, problems


    def split_p_difficulty(self, y_data):
        """Splits data based on success rate brackets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)."""
        # Calculate mean correctness for each problem
        mean_correct = y_data.mean(axis=1)  # num_problems x 1
        
        bin_slices = 1 / self.train_split_bins

        # Create 10 brackets from 0.0 to 1.0 with 0.1 increments
        bracket_edges = np.arange(0, 1.1, bin_slices)
        
        # Assign each problem to a bracket (0-9)
        # np.digitize with right=False puts values equal to bin edge in the higher bin
        # e.g., 0.1 goes in the 0.1-0.2 bracket, not 0.0-0.1
        assignments = np.digitize(mean_correct, bracket_edges, right=False) - 1
        
        # Clip to ensure we don't have any -1 values (for exact 0.0)
        assignments = np.clip(assignments, 0, len(bracket_edges)-2)

        # Create bracket labels for easier interpretation
        bracket_labels = [f"{bracket_edges[i]:.1f}-{bracket_edges[i+1]:.1f}" for i in range(len(bracket_edges)-1)]
        assignment_names = [bracket_labels[a] for a in assignments]
        
        num_problems = len(y_data)

        # Determine evaluation split explicitly
        eval_split = getattr(self, 'fixed_test_split', self.train_split) or self.train_split
        test_size = 1.0 - eval_split

        if test_size == 0:
            # No separate test set; test equals train
            train_idx = np.arange(num_problems)
            test_idx = np.arange(num_problems)
        elif test_size == 1.0:
            # No separate test set; test equals train
            train_idx = []
            test_idx = np.arange(num_problems)
        else:
            # Create fixed evaluation set based on eval_split
            # Use stratified sampling to maintain bracket distribution
            _, test_idx = train_test_split(np.arange(num_problems),
                                            test_size=test_size,
                                            random_state=self.random_seed,
                                            stratify=assignments)

            # If training on full data, train_idx is all data
            if self.train_split == 1.0:
                train_idx = np.arange(num_problems)
            else:
                # Else use train_idx as per self.train_split
                test_size = 1.0 - self.train_split
                train_idx, _ = train_test_split(
                    np.arange(num_problems),
                    test_size=test_size,
                    random_state=self.random_seed,
                    stratify=assignments,
                )


        # Handle subsets if specified
        # train_queries < 1: use fraction of self.train_split for training
        if self.train_queries < 1:
            num_train_problems = len(train_idx)
            new_num_train_problems = int(num_train_problems * self.train_queries)
            # Try to maintain stratification in the subset if possible
            try:
                subset_train_idx = train_test_split(
                    np.arange(num_train_problems),
                    train_size=new_num_train_problems,
                    random_state=self.random_seed,
                    stratify=assignments[train_idx]
                )[0]
            except ValueError:
                # Fall back to random selection if stratification fails
                subset_train_idx = self.rng.choice(num_train_problems, new_num_train_problems, replace=False)
            train_idx = train_idx[subset_train_idx]
        # train_queries > 1: use train_queries sampled from problems in (self.train_split)    
        elif self.train_queries > 1:
            num_train_problems = len(train_idx)
            train_idx = self.rng.choice(train_idx, min(self.train_queries, num_train_problems), replace=False)
        else:
            # shuffle the train when using all the train data:
            # train set will be shuffled when test_size is not 0
            if hasattr(self, 'shuffle_train_full') and self.shuffle_train_full:
                train_idx = self.rng.permutation(train_idx)

        # Make test set perfectly match train set if requested
        if getattr(self, "same_train_test", False):
            test_idx = train_idx
            # No need to shuffle the train set
            #train_idx = self.rng.choice(train_idx, self.train_queries, replace=False)

        # Print bracket distribution for train and test sets
        train_bracket_counts = np.bincount(assignments[train_idx], minlength=len(bracket_labels))
        test_bracket_counts = np.bincount(assignments[test_idx], minlength=len(bracket_labels))
        
        print("\nDifficulty bracket distribution:", flush=True)
        print(f"{'Bracket':<10} {'Train':<8} {'Test':<8}", flush=True)
        print("-" * 28, flush=True)
        for i, label in enumerate(bracket_labels):
            print(f"{label:<10} {train_bracket_counts[i]:<8} {test_bracket_counts[i]:<8}", flush=True)
        else:
            pass

        return (train_idx, test_idx, assignments)

    def split_training_data(self):
        """Prepares training and testing datasets with preprocessing and normalization."""
        X_data, y_data, verifier_names, extracted_answers, problems = self.setup()
        train_idx, test_idx, assignments = self.split_p_difficulty(y_data)

        # ------------------------------------------------------------------------------------------------
        # Split data into train and test
        train_data, test_data = X_data[train_idx], X_data[test_idx]
        y_train = y_data[train_idx]
        y_test = y_data[test_idx]

        train_answers = extracted_answers[train_idx]
        test_answers = extracted_answers[test_idx]

        train_problems = problems[train_idx]
        test_problems = problems[test_idx]

        # ------------------------------------------------------------------------------------------------            
        # Select subset of train samples
        if self.train_samples < 1:
            # use train_samples fraction of total train samples
            num_train_samples = train_data.shape[1]
            new_num_train_samples = int(num_train_samples * self.train_samples)
            
            if hasattr(self, 'use_first_n_samples') and self.use_first_n_samples:
                # Take the first N samples
                new_train_samples_indices = np.arange(new_num_train_samples)
            else:
                # Random selection (original behavior)
                new_train_samples_indices = np.random.choice(num_train_samples, new_num_train_samples, replace=False)
            
            train_data = train_data[:, new_train_samples_indices]
            train_answers = [[s[x] for x in new_train_samples_indices] for s in train_answers]
            y_train = y_train[:, new_train_samples_indices]
        elif self.train_samples > 1:
            # use train_samples number of total train samples
            num_train_samples = train_data.shape[1]
            
            if hasattr(self, 'use_first_n_samples') and self.use_first_n_samples:
                # Take the first N samples (limited by available samples)
                new_train_samples_indices = np.arange(min(self.train_samples, num_train_samples))
            else:
                # Random selection (original behavior)
                new_train_samples_indices = np.random.choice(num_train_samples, min(self.train_samples, num_train_samples), replace=False)
            
            train_data = train_data[:, new_train_samples_indices]
            train_answers = [[s[x] for x in new_train_samples_indices] for s in train_answers]
            y_train = y_train[:, new_train_samples_indices]

        else:
            if hasattr(self, 'shuffle_samples') and self.shuffle_samples:
                num_train_samples = train_data.shape[1]
                new_train_samples_indices = np.random.permutation(np.arange(num_train_samples))
                train_data = train_data[:, new_train_samples_indices]
                train_answers = [[s[x] for x in new_train_samples_indices] for s in train_answers]
                y_train = y_train[:, new_train_samples_indices]

        # ------------------------------------------------------------------------------------------------            
        # Select subset of test samples
        if hasattr(self, 'test_samples') and self.test_samples < 1:
            # use test_samples fraction of total test samples
            num_test_samples = test_data.shape[1]
            new_num_test_samples = int(num_test_samples * self.test_samples)
            new_test_samples_indices = np.random.choice(num_test_samples, new_num_test_samples, replace=False)
            test_data = test_data[:, new_test_samples_indices]
            test_answers = [[s[x] for x in new_test_samples_indices] for s in test_answers]
            y_test = y_test[:, new_test_samples_indices]
        elif hasattr(self, 'test_samples') and self.test_samples > 1:
            # use test_samples number of total test samples
            num_test_samples = test_data.shape[1]
            new_test_samples_indices = np.random.choice(num_test_samples, min(self.test_samples, num_test_samples), replace=False)
            test_data = test_data[:, new_test_samples_indices]
            test_answers = [[s[x] for x in new_test_samples_indices] for s in test_answers]
            y_test = y_test[:, new_test_samples_indices]
        else:
            if hasattr(self, 'shuffle_samples') and self.shuffle_samples:
                num_test_samples = test_data.shape[1]
                new_test_samples_indices = self.rng.permutation(np.arange(num_test_samples))
                test_data = test_data[:, new_test_samples_indices]
                test_answers = [[s[x] for x in new_test_samples_indices] for s in test_answers]
                y_test = y_test[:, new_test_samples_indices]


        # ------------------------------------------------------------------------------------------------
        # Normalize the data:
        # print("Raw verifier scores before normalization:", test_data)
        train_data, mask_problems_train, mask_verifiers_train = self.preprocess(train_data, verifier_names)
        
        test_data, mask_problems_test, mask_verifiers_test = self.preprocess(test_data, verifier_names)

        # Preprocessing we remove all the problems and verifiers with only nan responses
        y_train = y_train[mask_problems_train]
        y_test = y_test[mask_problems_test]

        if len(train_data) == 0:
            verifier_names = np.array(verifier_names)[mask_verifiers_test].tolist()
        else:
            verifier_names = np.array(verifier_names)[mask_verifiers_train].tolist()

        # ------------------------------------------------------------------------------------------------
        # Store data
        self.train_data = (train_data, y_train)
        self.test_data = (test_data, y_test)

        self.train_idx, self.test_idx, self.assignments = train_idx, test_idx, assignments
        self.verifier_names = verifier_names
        self.train_answers = train_answers
        self.test_answers = test_answers
        self.train_problems = train_problems
        self.test_problems = test_problems
        # ------------------------------------------------------------------------------------------------
        # Find closest train problem for each test problem
        self.find_closest_train_problem()

        self.drop_verifiers()

    def find_closest_train_problem(self):
            """
            Finds the closest problem in the train set for each problem in the test set.

            Args:
                X_train: (num_train_problems, num_samples, num_verifiers) - Train set
                X_test: (num_test_problems, num_samples, num_verifiers) - Test set

            Returns:
                distances: (num_test_problems, num_train_problems) - Distances between test problems and train problems
                closest_train_idx: (num_test_problems,) - Closest train problem indices for each test problem
            """
            X_train, _ = self.train_data
            X_test, _ = self.test_data

            if len(X_train) == 0:
                self.closest_train_idxs = np.arange(len(X_test))
                self.distances = np.zeros((len(X_test), 1))
                return

            if self.closest_train_problem_method == "mean_verifier_distance":
                # Aggregate each problem's samples into a single representation (e.g., mean across samples)
                X_train_repr = X_train.mean(axis=1)  # Shape: (num_train_problems, num_verifiers)
                X_test_repr = X_test.mean(axis=1)    # Shape: (num_test_problems, num_verifiers)
            elif self.closest_train_problem_method == "cov_verifier_distance":
                # 1) Compute the mean for each problem (train/test)
                X_train_mean = X_train.mean(axis=1)  # (num_train_problems, num_verifiers)
                X_test_mean = X_test.mean(axis=1)    # (num_test_problems, num_verifiers)
                # 2) Compute the sample covariance for each problem
                #    np.cov expects shape (num_verifiers, num_samples), hence we transpose each slice
                X_train_cov = np.array([np.cov(X_train[i].T) for i in range(len(X_train))]) # (num_train_problems, num_verifiers, num_verifiers)
                X_test_cov = np.array([np.cov(X_test[i].T) for i in range(len(X_test))]) # (num_train_problems, num_verifiers, num_verifiers)
                # Now we need a single "distance" between each pair (test_i, train_j).
                # We'll define one simple approach:
                #   distance = ||mu_i - mu_j||_2 + ||Sigma_i - Sigma_j||_F
                # where ||.||_F is the Frobenius norm.
            elif self.closest_train_problem_method == "SBERT":
                model_name = "nomic-ai/modernbert-embed-base" #all-mpnet-base-v2
                model = SentenceTransformer(model_name)
                print(f"Embedding training data with {model_name}...", flush=True)
                X_train_repr = model.encode(self.train_problems)
                print(f"Embedding test data with {model_name}...", flush=True)
                X_test_repr = model.encode(self.test_problems)
                print(X_train_repr.shape, X_test_repr.shape, flush=True)
            else:
                raise NotImplementedError(f"Unknown closest train problem method: {self.closest_train_problem_method}")
            if self.closest_train_problem_metric_type in ["euclidean", "cosine"]:
                # Compute pairwise distances between test problems and train problems
                print(f"Computing distances according to {self.closest_train_problem_metric_type} metric", flush=True)
                self.distances = cdist(X_test_repr, X_train_repr, metric=self.closest_train_problem_metric_type)  # Shape: (num_test_problems, num_train_problems)
            elif self.closest_train_problem_metric_type == "frobenius":
                # Fast but not including geometric information
                num_test = len(X_test)
                num_train = len(X_train)
                distances = np.zeros((num_test, num_train))

                for i in range(num_test):
                    for j in range(num_train):
                        mean_diff = X_test_mean[i] - X_train_mean[j]
                        cov_diff = X_test_cov[i] - X_train_cov[j]
                        # Euclidian norm of the means + Frobenius norm of the covariance difference
                        dist_ij = np.linalg.norm(mean_diff) + np.linalg.norm(cov_diff, ord="fro")
                        distances[i, j] = dist_ij
                self.distances = distances
            else:
                raise NotImplementedError(f"Unknown closest train problem metric type: {self.closest_train_problem_metric_type}")
            # Find the closest train problem for each test problem
            self.closest_train_idxs = np.argsort(self.distances, axis=1, kind="stable")  # Shape: (num_test_problems,)


    def extract_embeddings(self, model_name="nomic-ai/modernbert-embed-base"):
        model = SentenceTransformer(model_name)
        print(f"Embedding training data with {model_name}...", flush=True)
        self.X_train_repr = model.encode(self.train_problems)
        self.X_test_repr = model.encode(self.test_problems)
        all_train_answers_repr = np.zeros(len(self.train_problems))
        all_test_answers_repr = np.zeros(len(self.test_problems))
        self.X_train_answers_repr = np.array(all_train_answers_repr)
        self.X_test_answers_repr = np.array(all_test_answers_repr)


    def drop_verifiers(self):
        # TODO: first step towards merging verifier drop
        y_subset = self.train_data[1] # (num_train_problems, num_samples, num_verifiers)
        X_subset = self.train_data[0] # (num_train_problems, num_samples, num_verifiers)
        
        num_problems, num_samples, num_verifiers = X_subset.shape

        if len(X_subset) == 0:
            print("No train data to drop verifiers", flush=True)
            # TODO: can use even without labels
            return

        X_subset = X_subset.reshape(num_problems*num_samples, -1)
        y_subset = y_subset.reshape(num_problems*num_samples, -1)


        def _get_balanced_idxs(marginals, rule):
            """
            Discard verifiers based on the rule:

            The marginals are the mean of the votes for each verifier.
            If marginals ~ 0.5, the verifier is balanced.
            If marginals ~ 1.0, the verifier says correct almost all the time.
            If marginals ~ 0.0, the verifier says incorrect almost none of the time.

            The rule can be:
            - "all": keep all verifiers with marginals between 0.1 and 0.9
            - "small": keep only the verifiers with marginals < 0.5
            - "large": keep only the verifiers with marginals > 0.5

            Args:
                marginals: np.ndarray(num_verifiers,)
                rule: "all", "small", "large"
            Returns:
                balanced_idxs: np.ndarray(num_verifiers,)
            """
            if rule == "all":
                balanced_idxs = np.where((marginals > 0.1) & (marginals < 0.9))[0]
            elif rule == "small":
                balanced_idxs = np.where(marginals > 0.5)[0]
            elif rule == "large":
                balanced_idxs = np.where(marginals < 0.5)[0]
            else: # Use all indices if not specified
                balanced_idxs = np.arange(len(marginals))

            return balanced_idxs
        
        def _drop_deps(votes, truth, current_verifiers):
            """
            Discard verifiers based on dependencies.

            Select the verifiers that are maximally independent by minimizing the linear dependency among the outputs and the ground truth.

            Args:
                votes: np.ndarray(num_problems, num_samples, num_verifiers)
                truth: np.ndarray(num_problems, num_samples)
                current_verifiers: list(num_verifiers,)
            Returns:
                remaining_idxs: np.ndarray(num_verifiers,)
            """
            # Use drop_k from config if k is not provided
            k = min(self.max_num_independent_verifiers, len(current_verifiers))
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

            sorted_sparsity = {k: v for k, v in sorted(triple_to_sparsity.items(), key=lambda x: x[1])}

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

        if self.drop_imbalanced_verifiers is not None:
            if self.drop_imbalanced_verifiers == "adaptive":
                cb = y_subset.mean() # Use subset for class balance calculation
                if cb < 0.2:
                    rule = "large" # discard all the verifiers that vote mostly 'yes'
                elif cb > 0.8:
                    rule = "small" # discard all the verifiers that vote mostly 'no' if the cluster is easy 
                else:
                    rule = "all"
                print(f"Setting rule for discarding verifiers to be: {rule}.", flush=True)
            else:
                rule = self.drop_imbalanced_verifiers

            marginals = X_subset.mean(axis=0)  # Use subset for marginals calculation
            balanced_v_idxs = _get_balanced_idxs(marginals, rule)
            discarded_names = [v for i, v in enumerate(self.verifier_names) if i not in balanced_v_idxs]
            balanced_names = [v for i, v in enumerate(self.verifier_names) if i in balanced_v_idxs]
            print(f"\nDiscarding {len(discarded_names)} verifiers: \n{discarded_names}", flush=True)
            self.verifier_idxs = balanced_v_idxs

            train_data, y_train = self.train_data
            test_data, y_test = self.test_data
            verifier_names = self.verifier_names

            train_data = train_data[..., balanced_v_idxs]
            test_data = test_data[..., balanced_v_idxs]
            verifier_names = np.array(verifier_names)[balanced_v_idxs].tolist()

            
            self.train_data = (train_data, y_train)
            self.test_data = (test_data, y_test)

            self.verifier_names = verifier_names

            X_subset = X_subset[..., balanced_v_idxs]

        if self.use_deps == "drop":
            _, remaining_idxs = _drop_deps(X_subset, y_subset.squeeze(), current_verifiers=self.verifier_names)

            train_data, y_train = self.train_data
            test_data, y_test = self.test_data
            verifier_names = self.verifier_names

            train_data = train_data[..., remaining_idxs]
            test_data = test_data[..., remaining_idxs]
            verifier_names = np.array(verifier_names)[remaining_idxs].tolist()

            self.train_data = (train_data, y_train)
            self.test_data = (test_data, y_test)

            self.verifier_names = verifier_names
            
            print(f"Verifiers after dropping dependencies: {self.verifier_names}", flush=True)


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get indices of problems that have at least one positive sample
        self.positive_problem_indices = np.where(np.any(dataset.labels == 1, axis=1))[0]
        self.negative_problem_indices = np.where(np.all(dataset.labels == 0, axis=1))[0]
        
        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        
    def __iter__(self):
        if self.shuffle:
            # Shuffle indices
            np.random.shuffle(self.positive_problem_indices)
            np.random.shuffle(self.negative_problem_indices)
        
        # Create batches ensuring at least one problem with positive samples per batch
        for i in range(self.num_batches):
            # Always include at least one problem that has positive samples
            batch_indices = [self.positive_problem_indices[i % len(self.positive_problem_indices)]]
            
            # Fill rest of batch with random problems (can be positive or negative)
            remaining_size = self.batch_size - 1
            remaining_indices = np.random.choice(
                np.concatenate([self.positive_problem_indices, self.negative_problem_indices]),
                size=remaining_size,
                replace=False
            )
            
            batch_indices.extend(remaining_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

class BertEmbeddingDataset(Dataset):
    def __init__(self, dataset, mode="train", augmentation_config=None, exclude_all_zeros=False):
        self._dataset = dataset
        self.mode = mode
        self.augmentation_config = augmentation_config or {
            "verifier_noise": False,
            "verifier_dropout": False,
            "embedding_mixup": False,
            "verifier_noise_std": 0.1,
            "verifier_dropout_rate": 0.2,
            "mixup_alpha": 0.2,
            "balanced_batch": False  # New config for balanced batch sampling
        }
        self.exclude_all_zeros = exclude_all_zeros

        self.get_embeddings()
        
        # Store indices for balanced sampling
        if self.augmentation_config["balanced_batch"]:
            self.positive_problem_indices = np.where(np.any(self.labels == 1, axis=1))[0]
            self.negative_problem_indices = np.where(np.all(self.labels == 0, axis=1))[0]

    def get_embeddings(self):
        if self.mode == "train":
            # get embeddings for train problems
            # shape: ( num_train_problems, embedding_dim)
            self.embeddings = self._dataset.X_train_repr
            # get embeddings for train answers
            # shape: ( num_train_problems, num_samples, embedding_dim)
            self.answers = self._dataset.X_train_answers_repr
            # verifier scores for train problems
            # shape: ( num_train_problems, num_samples, num_verifiers)
            self.verifier_scores = self._dataset.train_data[0]
            # Labels for train problems
            # shape: ( num_train_problems, num_samples)
            self.labels = self._dataset.train_data[1]
            self.assignments = self._dataset.assignments[self._dataset.train_idx]

            # Filter out samples where all labels are 0 if exclude_all_zeros is True
            if self.exclude_all_zeros:
                mask = np.any(self.labels == 1, axis=1)
                self.embeddings = self.embeddings[mask]
                self.answers = self.answers[mask]
                self.verifier_scores = self.verifier_scores[mask]
                self.labels = self.labels[mask]
                self.assignments = self.assignments[mask]

        elif self.mode == "test":
            # get embeddings for test problems
            # shape: ( num_test_problems, embedding_dim)
            self.embeddings = self._dataset.X_test_repr
            # get embeddings for test answers
            # shape: ( num_test_problems, num_samples, embedding_dim)
            self.answers = self._dataset.X_test_answers_repr
            # verifier scores for test problems
            # shape: ( num_test_problems, num_samples, num_verifiers)
            self.verifier_scores = self._dataset.test_data[0]
            # Labels for test problems
            # shape: ( num_test_problems, num_samples)
            self.labels = self._dataset.test_data[1]
            self.assignments = self._dataset.assignments[self._dataset.test_idx]
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self):
        return len(self.embeddings)

    def _apply_verifier_noise(self, verifier_scores):
        if not self.augmentation_config["verifier_noise"]:
            return verifier_scores
        noise = torch.randn_like(verifier_scores) * self.augmentation_config["verifier_noise_std"]
        return verifier_scores + noise

    def _apply_verifier_dropout(self, verifier_scores):
        if not self.augmentation_config["verifier_dropout"]:
            return verifier_scores
        mask = torch.rand_like(verifier_scores) > self.augmentation_config["verifier_dropout_rate"]
        return verifier_scores * mask

    def _apply_embedding_mixup(self, embeddings, verifier_scores, labels):
        if not self.augmentation_config["embedding_mixup"]:
            return embeddings, verifier_scores, labels
        
        # Randomly select another sample
        idx2 = torch.randint(0, len(self.embeddings), (1,)).item()
        lambda_ = np.random.beta(self.augmentation_config["mixup_alpha"], 
                               self.augmentation_config["mixup_alpha"])
        
        # Mix embeddings
        mixed_embeddings = lambda_ * embeddings + (1 - lambda_) * self.embeddings[idx2]
        
        # Mix verifier scores
        mixed_verifier_scores = lambda_ * verifier_scores + (1 - lambda_) * self.verifier_scores[idx2]
        
        # Mix labels
        mixed_labels = lambda_ * labels + (1 - lambda_) * self.labels[idx2]
        
        return mixed_embeddings, mixed_verifier_scores, mixed_labels

    def __getitem__(self, idx=None):
        # Get for each problem in the dataset
        # embeddings from the question
        # embeddings from the answer
        # verifier scores
        # labels
        questions = self.embeddings[idx]
        answers = self.answers[idx]
        labels = self.labels[idx]
        verifier_scores = self.verifier_scores[idx]

        # Convert to tensors
        questions = torch.tensor(questions, dtype=torch.float32)
        answers = torch.tensor(answers, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        verifier_scores = torch.tensor(verifier_scores, dtype=torch.float32)

        # Apply augmentations during training
        if self.mode == "train":
            # Apply verifier noise
            verifier_scores = self._apply_verifier_noise(verifier_scores)
            
            # Apply verifier dropout
            verifier_scores = self._apply_verifier_dropout(verifier_scores)
            
            # Apply embedding mixup
            questions, verifier_scores, labels = self._apply_embedding_mixup(
                questions, verifier_scores, labels
            )

        return {
            "embedding": questions,
            "verifier_scores": verifier_scores,
            "labels": labels,
            "answers": answers,
            "assignments": torch.tensor(self.assignments[idx], dtype=torch.float32)
        }

    def get_balanced_batch_sampler(self, batch_size, shuffle=True):
        """Returns a balanced batch sampler that ensures at least one positive sample per batch."""
        return BalancedBatchSampler(self, batch_size, shuffle)
