import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import torch
import time
import h5py
import json
import numpy as np
from dataclasses import dataclass
import concurrent.futures
import multiprocessing
from rm_models import (
    GRMModel, SkyworksModel, URMModel, QRMModel, GPMModel,
    GRMLlama32Model, OffsetBiasModel, GRMGemmaModel, ArmorRMModel,
    QwenPRMModel, Qwen72BModel, EurusPRMStage1Model, EurusPRMStage2Model,
    INFORMModel, SkyworksGemmaModel,  QRMGemmaModel, LDLRewardGemmaModel,
    InternLM2RewardModel, InternLM2Reward7BModel, DecisionTreeRewardModel8B, 
    DecisionTreeRewardModel27B, Qwen72BPRMModel
)
from lm_judges import get_judge, JUDGE_REGISTRY
from weaver.dataset import create_df_from_h5s, create_df_from_h5

@dataclass
class ProcessingMetadata:
    """Metadata about the processing run"""
    timestamp: str
    dataset_path: str
    num_instructions: int
    num_samples_per_instruction: int
    max_input_length: int
    reward_models_used: List[str]
    lm_judges_used: List[str]  # Add this field
    processing_time: float
    gpu_allocations: Dict[str, int]

def get_available_gpus() -> List[int]:
    """Get list of available GPU indices"""
    return list(range(torch.cuda.device_count()))

def setup_logging() -> logging.Logger:
    """Configure basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("unified_judging")

def preprocess_h5_dataset(
    path: str,
    max_rows: Optional[int] = None,
    max_samples: Optional[int] = None,
    start_row: int = 0,
    end_row: Optional[int] = None,
    logger: logging.Logger = None
) -> Dataset:
    """Load and preprocess dataset from local path or HuggingFace hub"""
    try:
        # Load dataset
        logger.info(f"Loading dataset from {path}")
        dataset = create_df_from_h5(path, data_only=True)
        for key in ['samples', 'instruction', 'answer_correct']:
            logger.info(f"dataset[{key}]: {dataset[key]}")
                    
        # Validate required columns
        required_columns = {'instruction', 'samples'}
        missing = required_columns - set(dataset.keys())
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
            
        # Apply row range and limit if specified
        if end_row is not None:
            end_idx = min(end_row, len(dataset))
        else:
            end_idx = len(dataset)
            
        if start_row > 0 or end_row is not None:
            for key in ['samples', 'instruction', 'answer_correct']:
                dataset[key] = dataset[key][start_row:end_idx]
            # dataset = dataset.select(range(start_row, end_idx))
            logger.info(f"Processing rows {start_row} to {end_idx-1}")
            
        if max_rows is not None:
            dataset = dataset.head(max_rows)
            logger.info(f"Limited dataset to {len(dataset)} rows")
            
        # Apply sample limit if specified
        if max_samples is not None:
            dataset['samples'] = dataset['samples'].apply(lambda x: x[:max_samples])
            dataset['answer_correct'] = dataset['answer_correct'][start_row:end_idx]
            logger.info(f"Limited samples per instruction to {len(dataset['samples'][0])}")
            
        logger.info(
            f"Dataset loaded successfully:\n"
            f"- Number of instructions: {len(dataset['instruction'])}\n"
            f"- Samples per instruction: {len(dataset['samples'][0])}\n"
            # f"- Available columns: {list(dataset.keys())}"
        )
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

def preprocess_dataset(
    path: str,
    max_rows: Optional[int] = None,
    max_samples: Optional[int] = None,
    start_row: int = 0,
    end_row: Optional[int] = None,
    logger: logging.Logger = None
) -> Dataset:
    """Load and preprocess dataset from local path or HuggingFace hub"""
    try:
        # Load dataset
        logger.info(f"Loading dataset from {path}")
        try:
            if Path(path).exists():  # Local dataset
                try:
                    dataset = load_from_disk(path)
                    logger.info("Loaded local dataset")
                except Exception as e:
                    dataset = load_dataset(path)['train']
                    logger.info("Loaded local dataset")
            else:  # Try HuggingFace dataset
                try:
                    dataset = load_dataset(path)['test']
                    logger.info("Loaded HuggingFace dataset")
                except Exception as e:
                    dataset = load_dataset(path)['data']
                    logger.info("Loaded HuggingFace dataset")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {path}: {e}")
        
        # Handle column naming variants
        if 'problem' in dataset.features and 'instruction' not in dataset.features:
            logger.info("Found 'problem' column, renaming to 'instruction'")
            dataset = dataset.rename_column('problem', 'instruction')
            
        # Validate required columns
        required_columns = {'instruction', 'samples'}
        missing = required_columns - set(dataset.features.keys())
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
            
        # Apply row range and limit if specified
        if end_row is not None:
            end_idx = min(end_row, len(dataset))
        else:
            end_idx = len(dataset)
            
        if start_row > 0 or end_row is not None:
            dataset = dataset.select(range(start_row, end_idx))
            logger.info(f"Processing rows {start_row} to {end_idx-1}")
            
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} rows")
            
        # Apply sample limit if specified
        if max_samples is not None:
            dataset = dataset.map(
                lambda x: {'samples': x['samples'][:max_samples]},
                desc="Limiting samples per instruction"
            )
            logger.info(f"Limited samples per instruction to {max_samples}")
            
        logger.info(
            f"Dataset loaded successfully:\n"
            f"- Number of instructions: {len(dataset)}\n"
            f"- Samples per instruction: {len(dataset[0]['samples'])}\n"
            f"- Available columns: {list(dataset.features.keys())}"
        )
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

def post_process_dataset(
    dataset: Dataset,
    metadata: ProcessingMetadata,
    is_h5: bool,
    logger: logging.Logger
) -> Dataset:
    """Post-process dataset to ensure consistent format and add metadata"""
    try:
        logger.info("Starting post-processing...")
        
        # Validate scores
        def validate_row(example):
            num_samples = len(example['samples'])
            for model_name in metadata.reward_models_used:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    # Check all score types
                    for score_type in ['min_scores', 'max_scores', 'avg_scores']:
                        num_scores = len(example.get(f'{model_name}_{score_type}', []))
                        if num_samples != num_scores:
                            logger.warning(
                                f"Mismatch for {model_name} {score_type}: "
                                f"{num_samples} samples but {num_scores} scores"
                            )
                else:
                    num_scores = len(example.get(f'{model_name}_scores', []))
                    if num_samples != num_scores:
                        logger.warning(
                            f"Mismatch for {model_name}: "
                            f"{num_samples} samples but {num_scores} scores"
                        )
            return example

        def validate_row_h5(idx, num_samples):
            for model_name in metadata.reward_models_used:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    # Check all score types
                    for score_type in ['min_scores', 'max_scores', 'avg_scores']:
                        num_scores = len(dataset[f'{model_name}_{score_type}'][idx])
                        if num_samples != num_scores:
                            logger.warning(
                                f"Mismatch for {model_name} {score_type}: "
                                f"{num_samples} samples but {num_scores} scores"
                            )
                else:
                    num_scores = len(dataset[f'{model_name}_scores'][idx])
                    if num_samples != num_scores:
                        logger.warning(
                            f"Mismatch for {model_name}: "
                            f"{num_samples} samples but {num_scores} scores"
                        )

        if is_h5:
            for idx, sample in enumerate(dataset['samples']):
                num_samples = len(sample)
                validate_row_h5(idx, num_samples)
        else:
            dataset = dataset.map(validate_row)

        final_dataset = dataset
        
        logger.info("Post-processing complete")
        return final_dataset
        
    except Exception as e:
        logger.error(f"Error in post-processing: {e}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified RM and LM Judge Evaluation"
    )
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset or HuggingFace dataset name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed dataset (e.g., 'path/to/output.hf')"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="scenegen",
        help="Type of task to process (scenegen or math)"
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        help="Maximum number of instructions/rows to process"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum samples per instruction to process"
    )
    
    # Reward model arguments
    parser.add_argument(
        "--reward_models",
        type=str,
        default="",
        # default="GRM,Skyworks,URM,QRM,GPM,GRMLlama32,OffsetBias,GRMGemma,ArmorRM,QwenPRM,EurusPRMStage1,EurusPRMStage2,InternLM2Reward7B,DecisionTreeReward8B",
        help="Comma-separated list of reward models to use (e.g., 'GRM,Skyworks')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for reward model inference"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=8192,
        help="Maximum input length for reward model tokenization"
    )
    parser.add_argument(
        "--lm_judges",
        type=str,
        default="",
        help=f"Optional: Comma-separated list of LM judges to use. Available: {list(JUDGE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--verdicts_per_sample",
        type=int,
        default=1,
        help="Number of verdicts to collect per sample for LM judges (if using judges)"
    )
    
    # Add new argument for critique mode
    parser.add_argument(
        "--critique_mode",
        action="store_true",
        help="Use critique mode for LM judges"
    )
    
    # Add new argument for sequential processing
    parser.add_argument(
        "--sequential_rm_processing",
        action="store_true",
        help="Process reward models sequentially instead of in parallel"
    )
    
    # Add new arguments for row range
    parser.add_argument(
        "--start_row",
        type=int,
        help="Starting row index to process (inclusive)",
        default=0
    )
    parser.add_argument(
        "--end_row",
        type=int,
        help="Ending row index to process (exclusive). If not specified, processes until the end",
        default=None
    )
    
    parser.add_argument(
        "--push_to_hub",
        type=str,
        help="Push to HuggingFace Hub with this dataset name (e.g., 'hazyresearch/MMLU_Final')"
    )

    
    return parser.parse_args()

def process_with_reward_model(
    model_name: str,
    dataset: Dataset,
    gpu_idx: int,
    batch_size: int,
    max_input_length: int,
    is_h5: bool,
    logger: logging.Logger
) -> Dict[int, List[float]]:
    """Process dataset with a single reward model"""
    model = None
    try:
        device = f"cuda:{gpu_idx}"
        
        # Initialize appropriate model
        if model_name == "Qwen72B":
            model = Qwen72BModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "QwenPRM":
            model = QwenPRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
            scores_by_row = {}
            
            # Process all samples for each row
            if is_h5:
                enumerator = zip(dataset['instruction'], dataset['samples'])
            else:
                enumerator = dataset
            for idx, row in enumerate(enumerator):
                if is_h5:
                    instruction = row[0]
                    samples = row[1]
                else:
                    instruction = row['instruction']
                    samples = row['samples']
                
                try:
                    # Get scores for all samples
                    scores = model.get_scores([instruction] * len(samples), samples)
                    step_scores = model.get_step_scores()
                    
                    # Validate scores
                    if not any(scores.values()):  # Check if all score lists are empty
                        logger.warning(f"Row {idx}: QwenPRM returned empty scores")
                        scores = {
                            'min_scores': [None] * len(samples),
                            'max_scores': [None] * len(samples),
                            'avg_scores': [None] * len(samples)
                        }
                    
                    # Validate step scores
                    if not any(step_scores.values()):  # Check if all step score lists are empty
                        logger.warning(f"Row {idx}: QwenPRM returned empty step scores")
                        step_scores = {i: [None] for i in range(len(samples))}
                    
                    # Store all scores
                    scores_by_row[idx] = {
                        'min_scores': scores.get('min_scores', [None] * len(samples)),
                        'max_scores': scores.get('max_scores', [None] * len(samples)),
                        'avg_scores': scores.get('avg_scores', [None] * len(samples)),
                        'step_scores': [step_scores.get(i, [None]) for i in range(len(samples))]
                    }
                except Exception as e:
                    logger.error(f"Error processing row {idx} with QwenPRM: {e}")
                    scores_by_row[idx] = {
                        'min_scores': [None] * len(samples),
                        'max_scores': [None] * len(samples),
                        'avg_scores': [None] * len(samples),
                        'step_scores': [[None]] * len(samples)
                    }
            
            return scores_by_row
        
        elif model_name == "EurusPRMStage2":
            model = EurusPRMStage2Model(device=device, batch_size=batch_size, max_input_length=max_input_length)
            scores_by_row = {}
            
            # Process all samples for each row
            if is_h5:
                enumerator = zip(dataset['instruction'], dataset['samples'])
            else:
                enumerator = dataset
            for idx, row in enumerate(enumerator):
                if is_h5:
                    instruction = row[0]
                    samples = row[1]
                else:
                    instruction = row['instruction']
                    samples = row['samples']
                
                try:
                    # Get scores for all samples
                    scores = model.get_scores([instruction] * len(samples), samples)
                    step_scores = model.get_step_scores()
                    
                    # Store all scores
                    scores_by_row[idx] = {
                        'min_scores': scores.get('min_scores', [None] * len(samples)),
                        'max_scores': scores.get('max_scores', [None] * len(samples)),
                        'avg_scores': scores.get('avg_scores', [None] * len(samples)),
                        'step_scores': [step_scores.get(i, []) for i in range(len(samples))]
                    }
                except Exception as e:
                    logger.error(f"Error processing row {idx} with EurusPRMStage2: {e}")
                    scores_by_row[idx] = {
                        'min_scores': [None] * len(samples),
                        'max_scores': [None] * len(samples),
                        'avg_scores': [None] * len(samples),
                        'step_scores': [[]] * len(samples)
                    }
            
            return scores_by_row
        
        elif model_name == "EurusPRMStage1":
            model = EurusPRMStage1Model(device=device, batch_size=batch_size, max_input_length=max_input_length)
            scores_by_row = {}
            
            # Process all samples for each row
            if is_h5:
                enumerator = zip(dataset['instruction'], dataset['samples'])
            else:
                enumerator = dataset
            for idx, row in enumerate(enumerator):
                if is_h5:
                    instruction = row[0]
                    samples = row[1]
                else:
                    instruction = row['instruction']
                    samples = row['samples']
                
                try:
                    # Get scores for all samples
                    scores = model.get_scores([instruction] * len(samples), samples)
                    step_scores = model.get_step_scores()
                    
                    # Store all scores
                    scores_by_row[idx] = {
                        'min_scores': scores.get('min_scores', [None] * len(samples)),
                        'max_scores': scores.get('max_scores', [None] * len(samples)),
                        'avg_scores': scores.get('avg_scores', [None] * len(samples)),
                        'step_scores': [step_scores.get(i, []) for i in range(len(samples))]
                    }
                except Exception as e:
                    logger.error(f"Error processing row {idx} with EurusPRMStage1: {e}")
                    scores_by_row[idx] = {
                        'min_scores': [None] * len(samples),
                        'max_scores': [None] * len(samples),
                        'avg_scores': [None] * len(samples),
                        'step_scores': [[]] * len(samples)
                    }
            
            return scores_by_row
        
        elif model_name == "DecisionTreeReward8B":
            model = DecisionTreeRewardModel8B(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "DecisionTreeReward27B":
            model = DecisionTreeRewardModel27B(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "LDLRewardGemma":
            model = LDLRewardGemmaModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "InternLM2RewardModel":
            model = InternLM2RewardModel(device=device, batch_size=1, max_input_length=max_input_length) # batch_size=1 to handle token issues
        elif model_name == "InternLM2Reward7B":
            model = InternLM2Reward7BModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "SkyworksGemma":
            model = SkyworksGemmaModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "QRMGemma":
            model = QRMGemmaModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "INFORM":
            model = INFORMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "GRM":
            model = GRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "Skyworks":
            model = SkyworksModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "URM":
            model = URMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "QRM":
            model = QRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "GPM":
            model = GPMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "GRMLlama32":
            model = GRMLlama32Model(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "OffsetBias":
            model = OffsetBiasModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "GRMGemma":
            model = GRMGemmaModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "ArmorRM":
            model = ArmorRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "GRM":
            model = GRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        elif model_name == "Qwen72BPRM":
            model = Qwen72BPRMModel(device=device, batch_size=batch_size, max_input_length=max_input_length)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        logger.info(f"Processing with {model_name} on GPU {gpu_idx}")
        
        # For all other models, gather instruction-response pairs
        
        all_instructions = []
        all_responses = []
        sample_map = []
        
        if is_h5:
            enumerator = zip(dataset['instruction'], dataset['samples'])
        else:
            enumerator = dataset
        for idx, row in enumerate(enumerator):
            if is_h5:
                instruction = row[0]
                samples = row[1]
            else:
                instruction = row['instruction']
                samples = row['samples']
            
            for sample in samples:
                all_instructions.append(instruction)
                all_responses.append(sample)
                sample_map.append(idx)
                
        # Get scores and handle None values
        scores = model.get_scores(all_instructions, all_responses)
        scores = [0.0 if score is None else float(score) for score in scores]
        
        # Reorganize scores by row
        scores_by_row = {}
        for idx, score in zip(sample_map, scores):
            if idx not in scores_by_row:
                scores_by_row[idx] = []
            scores_by_row[idx].append(score)
            
        return scores_by_row
        
    except Exception as e:
        logger.error(f"Error processing with {model_name}: {e}")
        # Return None scores instead of raising
        if is_h5:
            enumerator = zip(dataset['instruction'], dataset['samples'])
            return {idx: [None] * len(row[1]) for idx, row in enumerate(enumerator)}
        else:
            enumerator = dataset
            return {idx: [None] * len(row['samples']) for idx, row in enumerate(enumerator)}
        
    finally:
        if model is not None:
            model.unload()

def process_with_reward_models_parallel(
    dataset: Dataset,
    model_names: List[str],
    gpu_allocations: Dict[str, int],
    batch_size: int,
    max_input_length: int,
    is_h5: bool,
    logger: logging.Logger
) -> Dataset:
    """Process dataset with multiple reward models in batches based on available GPUs"""
    try:
        logger.info("Starting parallel reward model processing...")
        
        # Get number of available GPUs
        num_gpus = len(set(gpu_allocations.values()))
        num_models = len(model_names)
        
        # Calculate number of batches needed
        batch_size = min(num_gpus, num_models)
        num_batches = (num_models + batch_size - 1) // batch_size  # Ceiling division
        
        logger.info(f"Processing {num_models} models in {num_batches} batches of {batch_size} models each")
        
        # Process models in batches
        all_scores = {}
        mp_context = multiprocessing.get_context('spawn')
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_models)
            batch_models = model_names[start_idx:end_idx]
            
            logger.info(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            logger.info(f"Models in this batch: {batch_models}")
            
            # Process current batch of models in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(batch_models),
                mp_context=mp_context
            ) as executor:
                future_to_model = {
                    executor.submit(
                        process_with_reward_model,
                        model_name,
                        dataset,
                        gpu_allocations[model_name],
                        batch_size,
                        max_input_length,
                        is_h5,
                        logger
                    ): model_name
                    for model_name in batch_models
                }
                
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        scores = future.result()
                        all_scores[model_name] = scores
                        logger.info(f"Completed processing {model_name}")
                    except Exception as e:
                        logger.error(f"Model {model_name} failed: {e}")
                        if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                            all_scores[model_name] = {
                                idx: {
                                    'min_scores': [None] * len(row['samples']),
                                    'max_scores': [None] * len(row['samples']),
                                    'avg_scores': [None] * len(row['samples']),
                                    'step_scores': [[]] * len(row['samples'])
                                }
                                for idx, row in enumerate(dataset)
                            }
                        else:
                            all_scores[model_name] = {
                                idx: [None] * len(row['samples'])
                                for idx, row in enumerate(dataset)
                            }
            
            # Clear GPU memory after each batch
            torch.cuda.empty_cache()
            logger.info(f"Completed batch {batch_idx + 1}/{num_batches}")
        
        # Add scores to dataset
        def add_model_scores(example, idx):
            for model_name in model_names:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    scores = all_scores[model_name].get(idx, {})
                    # Add each score type as a separate column
                    example[f'{model_name}_min_scores'] = scores.get('min_scores', [None] * len(example['samples']))
                    example[f'{model_name}_max_scores'] = scores.get('max_scores', [None] * len(example['samples']))
                    example[f'{model_name}_avg_scores'] = scores.get('avg_scores', [None] * len(example['samples']))
                    example[f'{model_name}_step_scores'] = scores.get('step_scores', [[]] * len(example['samples']))
                else:
                    example[f'{model_name}_scores'] = all_scores[model_name].get(idx, [None] * len(example['samples']))
            return example
            
        if is_h5:
            # For h5 datasets (dict-like), add columns directly
            for model_name in model_names:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    column_name = f'{model_name}_min_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_max_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_avg_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_step_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [[]] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                else:
                    column_name = f'{model_name}_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
        else:
            dataset = dataset.map(
                add_model_scores,
                with_indices=True,
                desc="Adding model scores"
            )
        
        logger.info("Completed reward model processing")
        return dataset
        
    except Exception as e:
        logger.error(f"Model {model_name} failed: {e}")
        if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
            all_scores[model_name] = {
                idx: {
                    'min_scores': [None] * len(row['samples']),
                    'max_scores': [None] * len(row['samples']),
                    'avg_scores': [None] * len(row['samples']),
                    'step_scores': [[]] * len(row['samples'])
                }
                for idx, row in enumerate(dataset)
            }
        else:
            all_scores[model_name] = {
                idx: [None] * len(row['samples'])
                for idx, row in enumerate(dataset)
            }

def allocate_models_to_gpus(
    model_names: List[str],
    available_gpus: List[int]
) -> Dict[str, int]:
    """Allocate each model to a GPU, cycling through available ones"""
    allocations = {}
    num_gpus = len(available_gpus)
    num_models = len(model_names)
    
    # Define models that need multiple GPUs
    multi_gpu_models = {
        "Qwen72B": 8,  # Needs 8 GPUs
        "INFORM": 8,  # Needs 8 GPUs
        "SkyworksGemma": 1,
        "QRMGemma": 1,
        "LDLRewardGemma": 1,
        "InternLM2RewardModel": 1
    }
    
    if num_models > num_gpus:
        logging.info(
            f"More models ({num_models}) than available GPUs ({num_gpus}). "
            f"Models will be processed in batches."
        )
    
    # Sort models to ensure consistent allocation
    sorted_models = sorted(model_names)
    
    # Allocate models to single GPUs
    for i, model_name in enumerate(sorted_models):
        if available_gpus:  # Only allocate if we have GPUs left
            gpu_idx = available_gpus[i % len(available_gpus)]
            allocations[model_name] = gpu_idx
        else:
            logging.warning(f"No GPUs available for model {model_name}")
            allocations[model_name] = "cpu"
    
    return allocations

def process_with_lm_judge(
    judge_name: str,
    dataset: Dataset,
    available_gpus: List[int],
    num_verdicts: int,
    batch_size: int,
    logger: logging.Logger,
    critique_mode: bool = False,
    is_h5: bool = False,
    task_type: str = "scenegen"
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[str]]]:
    """Process dataset with a single LM judge"""
    judge = None
    try:
        model_info = JUDGE_REGISTRY[judge_name]
        if model_info["provider"] in ['openai', 'anthropic']:
            logger.info(f"Processing with {judge_name} using API")
        else:
            logger.info(f"Processing with {judge_name} using {len(available_gpus)} GPUs")
            
        judge = get_judge(
            judge_name=judge_name,
            num_verdicts=num_verdicts,
            batch_size=batch_size,
            tensor_parallel_size=len(available_gpus) if not model_info["provider"] else 1
        )
        
        scores_by_row = {}
        raw_verdicts_by_row = {}
        
        # Set up enumerator based on dataset type
        if is_h5:
            # For h5 dataset, zip only the columns we need to avoid iterating over all columns
            enumerator = zip(dataset['instruction'], dataset['samples'])
        else:
            enumerator = dataset
            
        for idx, row in enumerate(enumerator):
            if is_h5:
                instruction = row[0]
                samples = row[1]
            else:
                instruction = row['instruction']
                samples = row['samples']
            
            logger.info(f"Processing row {idx} with {len(samples)} samples")
            
            # Get scores and raw verdicts for all samples in this row
            if critique_mode:
                scores_dict, raw_verdicts = judge.get_critique_scores([instruction] * len(samples), samples, task_type=task_type)
            else:
                scores_dict, raw_verdicts = judge.get_scores([instruction] * len(samples), samples, task_type=task_type)
                
            logger.info(f"Row {idx} scores: {scores_dict}")
            
            # Store both verdicts and raw text
            scores_by_row[str(idx)] = [scores_dict.get(i, [None]) for i in range(len(samples))]
            raw_verdicts_by_row[str(idx)] = [raw_verdicts.get(i, None) for i in range(len(samples))]
                
        return scores_by_row, raw_verdicts_by_row
        
    except Exception as e:
        logger.error(f"Error processing with {judge_name}: {e}")
        raise
        
    finally:
        if judge is not None:
            judge.unload()

def process_with_judges_sequential(
    dataset: Dataset,
    judge_names: List[str],
    available_gpus: List[int],
    num_verdicts: int,
    batch_size: int,
    logger: logging.Logger,
    critique_mode: bool = False,
    is_h5: bool = False,
    task_type: str = "scenegen"
) -> Dataset:
    """Process dataset with multiple LM judges sequentially"""
    try:
        logger.info("\nStarting sequential LM judge processing...")
        logger.info(f"Available GPUs for judges: {available_gpus}")
        logger.info(f"Using {'critique' if critique_mode else 'standard'} mode")
        
        # Process each judge sequentially
        for judge_name in judge_names:
            logger.info(f"\nProcessing with {judge_name}")
            
            try:
                # Clear GPU memory before starting new judge
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                scores, raw_verdicts = process_with_lm_judge(
                    judge_name=judge_name,
                    dataset=dataset,
                    available_gpus=available_gpus,
                    num_verdicts=num_verdicts,
                    batch_size=batch_size,
                    logger=logger,
                    critique_mode=critique_mode,
                    is_h5=is_h5,
                    task_type=task_type
                )
                
                # Add scores and raw verdicts to dataset
                def add_judge_scores(example, idx):
                    str_idx = str(idx)
                    mode_suffix = "_critique" if critique_mode else ""
                    if str_idx in scores:
                        example[f'{judge_name}{mode_suffix}_verdicts'] = scores[str_idx]
                        # example[f'{judge_name}{mode_suffix}_raw_verdicts_text'] = raw_verdicts[str_idx]
                    else:
                        example[f'{judge_name}{mode_suffix}_verdicts'] = [[None]] * len(example['samples'])
                        # example[f'{judge_name}{mode_suffix}_raw_verdicts_text'] = [None] * len(example['samples'])
                    return example
                
                if is_h5:
                    # For h5 datasets (dict-like), add columns directly
                    mode_suffix = "_critique" if critique_mode else ""
                    column_name = f'{judge_name}{mode_suffix}_verdicts'
                    
                    dataset[column_name] = [
                        scores.get(str(idx), [[None]] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                else:
                    dataset = dataset.map(
                        add_judge_scores,
                        with_indices=True,
                        desc=f"Adding {judge_name} scores and raw verdicts"
                    )
                
            except Exception as e:
                logger.error(f"Error processing {judge_name}: {e}")
                raise  # Fail fast on errors
        
        logger.info("Completed LM judge processing")
        return dataset
        
    except Exception as e:
        logger.error(f"Error in judge processing: {e}")
        raise

def process_with_reward_models_sequential(
    dataset: Dataset,
    model_names: List[str],
    gpu_allocations: Dict[str, int],
    batch_size: int,
    max_input_length: int,
    is_h5: bool,
    logger: logging.Logger
) -> Dataset:
    """Process dataset with multiple reward models sequentially"""
    try:
        logger.info("Starting sequential reward model processing...")
        
        all_scores = {}
        for model_name in model_names:
            logger.info(f"\nProcessing with {model_name}")
            
            try:
                # Clear GPU memory before starting new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                scores = process_with_reward_model(
                    model_name=model_name,
                    dataset=dataset,
                    gpu_idx=gpu_allocations[model_name],
                    batch_size=batch_size,
                    max_input_length=max_input_length,
                    is_h5=is_h5,
                    logger=logger
                )
                all_scores[model_name] = scores
                logger.info(f"Completed processing {model_name}")
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                # Handle failures gracefully
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    all_scores[model_name] = {
                        idx: {
                            'min_scores': [None] * len(row['samples']),
                            'max_scores': [None] * len(row['samples']),
                            'avg_scores': [None] * len(row['samples']),
                            'step_scores': [[]] * len(row['samples'])
                        }
                        for idx, row in enumerate(dataset)
                    }
                else:
                    all_scores[model_name] = {
                        idx: [None] * len(row['samples'])
                        for idx, row in enumerate(dataset)
                    }

        # Add scores to dataset
        def add_model_scores(example, idx):
            for model_name in model_names:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    scores = all_scores[model_name].get(idx, {})
                    example[f'{model_name}_min_scores'] = scores.get('min_scores', [None] * len(example['samples']))
                    example[f'{model_name}_max_scores'] = scores.get('max_scores', [None] * len(example['samples']))
                    example[f'{model_name}_avg_scores'] = scores.get('avg_scores', [None] * len(example['samples']))
                    example[f'{model_name}_step_scores'] = scores.get('step_scores', [[]] * len(example['samples']))
                else:
                    example[f'{model_name}_scores'] = all_scores[model_name].get(idx, [None] * len(example['samples']))
            return example
            
        if is_h5:
            # For h5 datasets (dict-like), add columns directly
            for model_name in model_names:
                if model_name in ["QwenPRM", "EurusPRMStage1", "EurusPRMStage2"]:
                    column_name = f'{model_name}_min_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_max_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_avg_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                    column_name = f'{model_name}_step_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [[]] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
                else:
                    column_name = f'{model_name}_scores'
                    dataset[column_name] = [
                        all_scores[model_name].get(idx, [None] * len(dataset['samples'][idx]))
                        for idx in range(len(dataset['instruction']))
                    ]
        else:
            dataset = dataset.map(
                add_model_scores,
                with_indices=True,
                desc="Adding model scores"
            )
        
        logger.info("Completed reward model processing")
        return dataset
        
    except Exception as e:
        logger.error(f"Error in reward model processing: {e}")
        raise

def main():
    """Main execution function"""
    args = parse_args()
    logger = setup_logging()
    start_time = time.time()
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_h5 = args.dataset_path.endswith(".h5")
    
    try:
        # Parse reward models and judges
        model_names = [name.strip() for name in args.reward_models.split(",")]
        # Filter out any models that are empty strings
        model_names = [name for name in model_names if name]

        if len(model_names) == 0:
            logger.info("No reward models specified, skipping reward model processing")

        judge_names = []
        if args.lm_judges:
            judge_names = [name.strip() for name in args.lm_judges.split(",")]
            logger.info(f"Will process with LM judges: {judge_names}")
            if args.critique_mode:
                logger.info("Using critique mode for LM judges")
        
        # Get available GPUs
        available_gpus = get_available_gpus()
        if not available_gpus:
            logger.info("No GPUs available for local models")
            available_gpus = [0]
        
        # First process reward models
        logger.info("\nProcessing reward models...")
        gpu_allocations = allocate_models_to_gpus(model_names, available_gpus)
        logger.info(f"GPU allocations for reward models: {gpu_allocations}")
        
        if is_h5:
            dataset = preprocess_h5_dataset(
                path=args.dataset_path,
                max_rows=args.max_rows,
                max_samples=args.max_samples,
                start_row=args.start_row,
                end_row=args.end_row,
                logger=logger
            )
        else:
            dataset = preprocess_dataset(
                path=args.dataset_path,
                max_rows=args.max_rows,
                max_samples=args.max_samples,
                start_row=args.start_row,
                end_row=args.end_row,
                logger=logger
            )
        
        if len(model_names) > 0:
            if args.sequential_rm_processing:
                logger.info("Using sequential reward model processing")
                dataset = process_with_reward_models_sequential(
                    dataset=dataset,
                    model_names=model_names,
                    gpu_allocations=gpu_allocations,
                    batch_size=args.batch_size,
                    max_input_length=args.max_input_length,
                    is_h5=is_h5,
                    logger=logger
                )
            else:
                logger.info(f"Using parallel reward model processing, is_h5: {is_h5}")
                dataset = process_with_reward_models_parallel(
                    dataset=dataset,
                    model_names=model_names,
                    gpu_allocations=gpu_allocations,
                    batch_size=args.batch_size,
                    max_input_length=args.max_input_length,
                    is_h5=is_h5,
                    logger=logger
                )
        
        # Then process LM judges sequentially if specified
        if judge_names:
            logger.info("\nStarting LM judge processing...")
            dataset = process_with_judges_sequential(
                dataset=dataset,
                judge_names=judge_names,
                available_gpus=available_gpus,
                num_verdicts=args.verdicts_per_sample,
                batch_size=args.batch_size,
                logger=logger,
                critique_mode=args.critique_mode,
                is_h5=is_h5
            )
        
        # Create metadata
        metadata = ProcessingMetadata(
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
            dataset_path=args.dataset_path,
            num_instructions=len(dataset),
            num_samples_per_instruction=len(dataset['samples'][0]) if is_h5 else len(dataset[0]['samples']),
            max_input_length=args.max_input_length,
            reward_models_used=model_names,
            lm_judges_used=judge_names,
            processing_time=time.time() - start_time,
            gpu_allocations=gpu_allocations
        )
        
        # Save final dataset
        print("dataset answer_correct:", dataset['answer_correct'])
        final_dataset = post_process_dataset(dataset, metadata, is_h5, logger)
        print("final_dataset:", final_dataset.keys())
        if is_h5:
            with h5py.File(args.output_path, 'a') as h5f:
                # Save `instruction` as a simple string column
                instruction_arr = np.array(final_dataset["instruction"], dtype='S')
                if 'instruction' in h5f:
                    del h5f['instruction']
                h5f.create_dataset('instruction', data=instruction_arr, dtype=h5py.special_dtype(vlen=str))

                # Save `samples` as a ragged array of strings (variable-length)
                dt_samples = h5py.special_dtype(vlen=str)
                samples_arr = np.array([np.array([str(s) for s in sample], dtype=object) 
                                        for sample in final_dataset["samples"]], dtype=object)
                if 'samples' in h5f:
                    del h5f['samples']
                h5f.create_dataset('samples', data=samples_arr, dtype=dt_samples)
                
                # Save `answer_correct` as an n_tasks x n_samples boolean array
                answer_correct_arr = np.array(final_dataset["answer_correct"].tolist(), dtype=bool)
                logger.info(f"answer_correct_arr.shape: {answer_correct_arr.shape}")
                if 'answer_correct' in h5f:
                    del h5f['answer_correct']
                h5f.create_dataset('answer_correct', data=answer_correct_arr, dtype='bool')

                # Create verifier group for all score/verdict data
                if 'verifier' in h5f:
                    del h5f['verifier']
                verifier_grp = h5f.create_group('verifier')
                
                for key in final_dataset.keys():
                    if key not in ['instruction', 'samples', 'answer_correct']:
                        # If verdicts_arr is a pandas Series, convert it to a numpy array suitable for h5py
                        verdicts_arr = np.array(final_dataset[key].tolist())
                        verifier_grp.create_dataset(key, data=verdicts_arr, dtype='float')
        else:
            DatasetDict({"data": final_dataset}).save_to_disk(args.output_path)
        logger.info(f"Dataset saved locally to: {args.output_path}")
        
        # Push to hub if specified
        if args.push_to_hub:
            try:
                DatasetDict({"data": final_dataset}).push_to_hub(args.push_to_hub, private=True)
                logger.info(f"Successfully pushed to hub: {args.push_to_hub}")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")
                logger.info("Dataset still saved locally")
        
        # Print sample results
        logger.info("\nExample results:")
        if is_h5:
            example = {key: final_dataset[key][0] for key in final_dataset.keys()}
        else:
            example = final_dataset[0]
        logger.info(f"Instruction: {example['instruction']}")
        logger.info(f"Number of samples: {len(example['samples'])}")
        
        # Print scores
        logger.info("First sample scores:")
        for model_name in model_names:
            scores = example.get(f'{model_name}_scores', [])
            if scores and len(scores) > 0:
                score = scores[0]
                if score is not None:
                    logger.info(f"  {model_name}: {score:.3f}")
                else:
                    logger.info(f"  {model_name}: None")
        
        for judge_name in judge_names:
            scores = example.get(f'{judge_name}_scores', [])
            if scores and len(scores) > 0:
                score = scores[0]
                if score is not None:
                    logger.info(f"  {judge_name}: {score:.3f}")
                else:
                    logger.info(f"  {judge_name}: None")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
