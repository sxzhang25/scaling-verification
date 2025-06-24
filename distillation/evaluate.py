from sentence_transformers.cross_encoder import CrossEncoder
from datasets import load_dataset
import logging
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List
import argparse
import torch
from train import CustomCrossEncoder, TrainingConfig
import os
from tabulate import tabulate

# Disable PyTorch 2.0 Compiler optimizations
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0

# Setup logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a cross-encoder model')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True,
                      help='Base model name (e.g., answerdotai/ModernBERT-large)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='HuggingFace dataset path')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--start_row', type=int, default=0,
                      help='Starting row index for evaluation (inclusive)')
    parser.add_argument('--end_row', type=int, default=None,
                      help='Ending row index for evaluation (exclusive)')
    parser.add_argument('--only_solvable_queries', action='store_true', default=False,
                      help='Only evaluate on queries that have at least one correct answer')
    
    args = parser.parse_args()
    return args

def calculate_selection_at_1(predictions, labels, num_samples_per_row, only_solvable=False):
    """Calculate Selection@1 metric."""
    num_rows = len(predictions) // num_samples_per_row
    correct = 0
    total = 0
    
    for row_idx in range(num_rows):
        start_idx = row_idx * num_samples_per_row
        end_idx = start_idx + num_samples_per_row
        
        row_predictions = predictions[start_idx:end_idx]
        row_labels = labels[start_idx:end_idx]
        
        if only_solvable and sum(row_labels) == 0:  # Skip rows with no correct answers
            continue
            
        max_score_idx = np.argmax(row_predictions)
        if row_labels[max_score_idx] == 1:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def calculate_first_sample_accuracy(labels, num_samples_per_row, only_solvable=False):
    """Calculate accuracy of randomly selecting the first sample."""
    num_rows = len(labels) // num_samples_per_row
    correct = 0
    total = 0
    
    for row_idx in range(num_rows):
        start_idx = row_idx * num_samples_per_row
        end_idx = start_idx + num_samples_per_row
        
        row_labels = labels[start_idx:end_idx]
        
        if only_solvable and sum(row_labels) == 0:  # Skip rows with no correct answers
            continue
            
        # First sample is always at index 0
        if row_labels[0] == 1:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def evaluate_majority_voting(dataset, only_solvable=False):
    """Calculate Majority@1 metric."""
    total_problems = len(dataset['extracted_answers'])
    correct = 0
    total = 0
    
    for problem_idx in range(total_problems):
        extracted_answers = dataset['extracted_answers'][problem_idx]
        answer_correct = dataset['answer_correct'][problem_idx]
        
        if only_solvable and sum(answer_correct) == 0:  # Skip rows with no correct answers
            continue
            
        # Count occurrences of each answer
        answer_counts = Counter(extracted_answers)
        
        # Get most common answer
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Find indices of most common answer
        indices = [i for i, x in enumerate(extracted_answers) if x == most_common_answer]
        correct_count = sum(answer_correct[i] for i in indices)
        
        if correct_count > len(indices) / 2:
            correct += 1
        total += 1
    
    return correct, total

def calculate_weaver_scores_accuracy(data, only_solvable=False):
    """Calculate accuracy using Weaver scores by selecting highest scoring samples."""
    num_rows = len(data['samples'])
    correct = 0
    total = 0
    
    for row_idx in range(num_rows):
        weaver_scores = np.array(data['weaver_scores'][row_idx], dtype=np.float64)  # Use float64 for better precision
        labels = data['answer_correct'][row_idx]
        
        if only_solvable and sum(labels) == 0:  # Skip rows with no correct answers
            continue
            
        # Get the index of the sample with highest probability
        best_sample_idx = np.argmax(weaver_scores)
        if labels[best_sample_idx]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()
    
    # Create config for model loading
    config = TrainingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        mlp_hidden_dims=[1536, 768, 384] if "Qwen2" in args.model_name else [1024, 512, 256],
        dataset_path=args.dataset_path,
        seed=42
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = CustomCrossEncoder(config)
    if '/' in args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        # If checkpoint_path is a remote HF repository
        logger.info("Loading from HuggingFace repository...")
        model.load_finetuned_checkpoint(args.checkpoint_path)
    else:
        # If checkpoint_path is a local file
        model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    data = dataset["data"]
    
    # Calculate row range
    start_idx = args.start_row
    end_idx = args.end_row if args.end_row is not None else len(data)
    
    if start_idx > 0 or end_idx < len(data):
        logger.info(f"Evaluating rows from index {start_idx} to {end_idx}")
        data = data.select(range(start_idx, end_idx))
    
    # Create evaluation samples
    eval_samples = []
    binary_labels = []
    
    # Process dataset
    for idx in range(len(data)):
        instruction = data['instruction'][idx]
        samples = data['samples'][idx]
        labels = data['answer_correct'][idx]
        
        for sample, label in zip(samples, labels):
            eval_samples.append([instruction, sample])
            binary_labels.append(1 if label else 0)
    
    # Get model predictions
    predictions = []
    batch_size = args.batch_size
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_samples), batch_size), desc="Getting predictions"):
            batch_samples = eval_samples[i:i + batch_size]
            
            # Tokenize batch
            encoded = model.tokenizer(
                text=[sample[0] for sample in batch_samples],
                text_pair=[sample[1] for sample in batch_samples],
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Get predictions
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                outputs = model(input_ids, attention_mask)
                batch_predictions = outputs.float().cpu().numpy()
                predictions.extend(batch_predictions)
    
    # Calculate metrics
    num_samples_per_row = len(data['samples'][0])
    selection_at_1 = calculate_selection_at_1(predictions, binary_labels, num_samples_per_row, args.only_solvable_queries)
    first_sample_accuracy = calculate_first_sample_accuracy(binary_labels, num_samples_per_row, args.only_solvable_queries)
    majority_correct, total_problems = evaluate_majority_voting(data, args.only_solvable_queries)
    weaver_accuracy = calculate_weaver_scores_accuracy(data, args.only_solvable_queries)
    
    # Count valid rows
    num_rows = len(predictions) // num_samples_per_row
    valid_rows = sum(1 for i in range(num_rows) 
                    if sum(binary_labels[i*num_samples_per_row:(i+1)*num_samples_per_row]) > 0)
    
    # Create results table
    results = [
        ["Distilled Weaver - Selection@1", f"{selection_at_1*100:.2f}%"],
        ["FirstSample", f"{first_sample_accuracy*100:.2f}%"],
        ["Majority@1", f"{majority_correct/total_problems*100:.2f}%"],
        ["Full Weaver - Selection@1", f"{weaver_accuracy*100:.2f}%"],
        ["Total Answerable Rows", str(valid_rows)],
        ["Total Rows", str(num_rows)]
    ]
    
    # Print results in a nice table
    print("\nEvaluation Results:")
    print(tabulate(results, tablefmt="grid"))

if __name__ == "__main__":
    main()
