import os
import gc
import re
import argparse
import openai
import time
import subprocess
import openai
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from pathlib import Path

class DatasetHandler(ABC):
    """Base class for dataset handlers"""
    
    @abstractmethod
    def load_dataset(self, input_path: str) -> Any:
        """Load dataset from path"""
        pass
        
    @abstractmethod
    def extract_answer(self, problem: str, sample: str, client: openai.OpenAI, **kwargs) -> Optional[str]:
        """Extract and normalize answer from sample"""
        pass
        
    @abstractmethod
    def verify_answer(self, extracted: str, ground_truth: str) -> bool:
        """Verify if extracted answer matches ground truth"""
        pass
        
    def save_dataset(self, dataset: Any, output_path: str):
        """
        Save dataset to disk, handling overwrites safely
        TODO: the temp .bak file is not getting deleted properly
        """
        output_path = Path(output_path)
        
        if output_path.exists():
            # Force garbage collection to release file handles
            gc.collect()
            
            # Create temporary directory for new dataset
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "temp_dataset"
                dataset.save_to_disk(str(temp_path))
                
                # Move old dataset to backup, then copy new one
                backup_path = output_path.with_suffix('.bak')
                if backup_path.exists():
                    shutil.rmtree(backup_path, ignore_errors=True)
                shutil.move(str(output_path), str(backup_path))
                shutil.copytree(str(temp_path), str(output_path))
                
            subprocess.run(['rm', '-rf', str(backup_path)])
        else:
            dataset.save_to_disk(str(output_path))

    def _truncate_sample_columns(self, dataset: Any, max_samples: int) -> Any:
        """Truncate all sample-related columns to max_samples length"""
        if max_samples is None:
            return dataset
            
        def truncate_list(lst, max_len):
            return lst[:max_len] if lst is not None else lst
            
        # Get all columns that need truncation
        sample_columns = ['samples']
        score_columns = [col for col in dataset.column_names if '_scores' in col]
        verdict_columns = [col for col in dataset.column_names if '_verdicts' in col or '_correct' in col]
        
        # Create new columns dict
        new_columns = {}
        
        # Truncate samples and related columns (like gpm_scores)
        for col in sample_columns + score_columns:
            if col in dataset.column_names:
                new_columns[col] = [truncate_list(row, max_samples) for row in dataset[col]]
                
        # Truncate verdict columns (they should match extracted_answers length)
        for col in verdict_columns + ['extracted_answers']:
            if col in dataset.column_names:
                new_columns[col] = [truncate_list(row, max_samples) for row in dataset[col]]
        
        # Update dataset with truncated columns
        for col, values in new_columns.items():
            dataset = dataset.remove_columns(col)
            dataset = dataset.add_column(col, values)
            
        return dataset

class GPQAHandler(DatasetHandler):
    """Handler for GPQA multiple choice datasets"""
    
    def load_dataset(self, input_path: str) -> Any:
        try:
            dataset = load_from_disk(input_path)
        except:
            try:
                dataset = load_dataset(input_path)["data"]
            except:
                dataset = load_dataset(input_path)
        return dataset
    
    def extract_answer(self, problem: str, sample: str, client: openai.OpenAI, **kwargs) -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": "You are an answer extraction assistant. Extract the final answer from the response and return ONLY a single letter A-D. If no valid answer is found, return 'NONE'."
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\nResponse: {sample}\n\nExtract the final answer letter (A-D):"
            }
        ]
        
        for sleep_time in [1, 2, 4, 8]:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0
                )
                extracted = completion.choices[0].message.content.strip()
                logger.debug(f"Raw extraction: {extracted}")
                
                # Extract just the letter if it's in "The answer is: X" format
                answer_match = re.search(r'(?:the answer is:?\s*)?([A-D])', extracted, re.IGNORECASE)
                if answer_match:
                    extracted = answer_match.group(1)
                
                # Normalize to single letter
                if extracted and len(extracted) == 1 and extracted.upper() in 'ABCD':
                    normalized = extracted.upper()
                    logger.info(f"Normalized answer: {normalized}")
                    return normalized
                logger.debug(f"Invalid answer format: {extracted}")
                return None
                
            except Exception as e:
                logger.error(f"Error in extraction: {str(e)}")
                time.sleep(sleep_time)
        return None
    
    def verify_answer(self, extracted: str, ground_truth: str) -> bool:
        if not extracted:
            return False
            
        # Normalize both to single letters
        extracted_norm = extracted.strip().upper()
        ground_truth_norm = ground_truth.strip().upper()
        
        logger.debug(f"Verification: {extracted} → {extracted_norm} vs {ground_truth} → {ground_truth_norm}")
        return extracted_norm == ground_truth_norm

class MathAnswer(BaseModel):
    """Structured output for math answer extraction"""
    value: str  # The extracted numerical/mathematical answer
    confidence: float  # Confidence score between 0-1
    explanation: Optional[str] = None  # Optional explanation of extraction

class MathVerification(BaseModel):
    """Structured output for math answer verification"""
    is_correct: bool
    confidence: float
    explanation: Optional[str] = None

class MathHandler(DatasetHandler):
    """Handler for MATH datasets"""
    
    def __init__(self):
        self.extract_prompt = r"""Extract the mathematical answer from the solution. The answer will typically be inside a \boxed{} command.
If there are multiple boxed expressions, extract the final one. Return only the mathematical expression without any surrounding text.

Example 1:
Input: Therefore, $x = \boxed{5}$ is the solution.
Output: 5

Example 2:
Input: The final answer is $\boxed{\frac{\sqrt{3}}{2}}$.
Output: \frac{\sqrt{3}}{2}

Example 3:
Input: We get $\boxed{x = 2}$ and $\boxed{y = 3}$, so $\boxed{x + y = 5}$.
Output: 5"""

        self.verify_prompt = r"""Compare these two mathematical solutions and determine if they are equivalent. Focus on:
1. The final numerical or mathematical answer (typically in a \boxed{} command)
2. Mathematical equivalence (e.g., 1/2 = 0.5 = \frac{1}{2})
3. Different but valid solution methods that arrive at the same result

Return true only if the solutions are mathematically equivalent."""

    def extract_answer(self, problem: str, sample: str, client: openai.OpenAI, **kwargs) -> Optional[str]:
        """Extract answer using structured output parsing"""
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.extract_prompt},
                    {"role": "user", "content": sample}
                ],
                response_format=MathAnswer
            )
            result = completion.choices[0].message.parsed
            
            logger.debug(f"Raw extraction: {result.value}")
            
            # Normalize the answer
            normalized = self.normalize_answer(result.value)
            logger.info(f"Normalized answer: {normalized}")
            
            if result.confidence < 0.5:
                logger.warning(f"Low confidence answer extraction: {result.confidence}")
                if result.explanation:
                    logger.warning(f"Explanation: {result.explanation}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return None

    def verify_answer(self, sample: str, solution: str, client: openai.OpenAI = None) -> bool:
        """Verify answer by comparing full solution text"""
        try:
            if client is None:
                # Fall back to basic string comparison if no client provided
                return self.basic_verify(sample, solution)
                
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.verify_prompt},
                    {"role": "user", "content": f"Solution 1:\n{sample}\n\nSolution 2:\n{solution}"}
                ],
                response_format=MathVerification
            )
            result = completion.choices[0].message.parsed
            
            logger.debug("Comparing answers:")
            logger.debug(f"  Extracted: {sample}")
            logger.debug(f"  Ground truth: {solution}")
            logger.debug(f"  Verification result: {result.is_correct}")
            
            if result.confidence < 0.8:
                logger.warning(f"Low confidence verification: {result.confidence}")
                if result.explanation:
                    logger.warning(f"Explanation: {result.explanation}")
            
            return result.is_correct
            
        except Exception as e:
            logger.error(f"Error verifying answer: {e}")
            return False

    def basic_verify(self, sample: str, solution: str) -> bool:
        """Basic verification without LLM"""
        sample_norm = self.normalize_answer(sample)
        solution_norm = self.normalize_answer(solution)
        return sample_norm == solution_norm

    def normalize_answer(self, answer: str) -> str:
        """Normalize the extracted answer format"""
        if answer is None:
            return None
            
        # Remove any "The answer is:" prefix
        if isinstance(answer, str) and answer.lower().startswith("the answer is:"):
            answer = answer[14:].strip()
            
        # Remove any surrounding whitespace
        answer = answer.strip()
        
        # Remove any outer \boxed{} command
        boxed_match = re.match(r'\\boxed\{(.*)\}', answer)
        if boxed_match:
            answer = boxed_match.group(1)
            
        # Normalize fractions
        answer = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', answer)
        
        # Normalize square roots
        answer = re.sub(r'\\sqrt\{(\d+)\}', r'√\1', answer)
        
        return answer

    def load_dataset(self, path: str) -> Dataset:
        """Load dataset from file"""
        try:
            dataset = load_from_disk(path)
        except:
            try:
                dataset = load_dataset(path)['data']
            except:
                dataset = load_from_disk(path)['data']
        return dataset

    def _truncate_sample_columns(self, dataset: Dataset, max_samples: Optional[int] = None) -> Dataset:
        """Truncate sample-related columns if needed"""
        if max_samples is None:
            return dataset
            
        if 'samples' in dataset.column_names:
            dataset = dataset.map(
                lambda x: {'samples': x['samples'][:max_samples]},
                remove_columns=['samples']
            )
            
        return dataset

class MMLUHandler(DatasetHandler):
    """Handler for MMLU datasets with A-D multiple choice answers"""
    
    def load_dataset(self, input_path: str) -> Any:
        try:
            dataset = load_from_disk(input_path)
        except:
            dataset = load_dataset(input_path)['data']
        # Ensure required columns exist
        required_columns = ['instruction', 'samples', 'answer']
        missing = [col for col in required_columns if col not in dataset.column_names]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        return dataset
    
    def extract_answer(self, problem: str, sample: str, client: openai.OpenAI, **kwargs) -> Optional[str]:
        messages = [
            {
                "role": "system",
                "content": "You are an answer extraction assistant. Extract the final answer from the response and return ONLY a single letter A-D. Just return the letter, don't include any other text. If no valid answer is found, return 'NONE'."
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\nResponse: {sample}\n\nExtract the final answer letter (A-D):"
            }
        ]
        
        for sleep_time in [1, 2, 4, 8]:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0
                )
                extracted = completion.choices[0].message.content.strip()
                logger.debug(f"Raw extraction: {extracted}")
                
                # Extract just the letter if it's in "The answer is: X" format
                answer_match = re.search(r'(?:the answer is:?\s*)?([A-D])', extracted, re.IGNORECASE)
                if answer_match:
                    extracted = answer_match.group(1)
                
                # Normalize to single letter
                if extracted and len(extracted) == 1 and extracted.upper() in 'ABCD':
                    normalized = extracted.upper()
                    logger.info(f"Normalized answer: {normalized}")
                    return normalized
                logger.debug(f"Invalid answer format: {extracted}")
                return None
                
            except Exception as e:
                logger.error(f"Error in extraction: {str(e)}")
                time.sleep(sleep_time)
        return None
    
    def verify_answer(self, extracted: str, ground_truth: str) -> bool:
        if not extracted:
            logger.debug("Verification failed: extracted answer is None")
            return False
                
        # Normalize both to single letters
        extracted_norm = extracted.strip().upper()
        ground_truth_norm = ground_truth.strip().upper()
        
        logger.debug(f"Verification: {extracted} → {extracted_norm} vs {ground_truth} → {ground_truth_norm}")
        return extracted_norm == ground_truth_norm

class MMLUProHandler(MMLUHandler):
    """Handler for MMLU-Pro datasets with flexible multiple choice answers"""
    
    def load_dataset(self, input_path: str) -> Any:
        """Load dataset and verify required columns exist"""
        try:
            dataset = load_from_disk(input_path)
        except:
            dataset = load_dataset(input_path)['data']
            
        # Verify required columns
        required_columns = ['instruction', 'samples', 'answer', 'options']
        missing = [col for col in required_columns if col not in dataset.column_names]
        if missing:
            raise ValueError(f"Dataset missing required columns for MMLU-Pro: {missing}")
        return dataset
    
    def extract_answer(self, problem: str, sample: str, client: openai.OpenAI, **kwargs) -> Optional[str]:
        # Get the options for this specific question
        options = kwargs.get('options')
        if options is None:
            raise ValueError("MMLU-Pro handler requires 'options' in kwargs to determine valid answer range")
            
        num_options = len(options)
        if num_options == 0:
            raise ValueError("MMLU-Pro question has empty options list")
            
        valid_answers = set(chr(i) for i in range(ord('A'), ord('A') + num_options))
        
        messages = [
            {
                "role": "system",
                "content": f"You are an answer extraction assistant. Extract the final multiple choice answer from the response. Return ONLY a single letter (A-{chr(ord('A') + num_options - 1)}). If no valid answer letter is found, return 'NONE'."
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\nResponse: {sample}\n\nExtract the final answer letter:"
            }
        ]
        
        for sleep_time in [1, 2, 4, 8]:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0
                )
                extracted = completion.choices[0].message.content.strip().upper()
                logger.debug(f"Raw extraction: {extracted}")
                
                # Extract just the letter if it's in "The answer is: X" format
                answer_match = re.search(r'(?:THE ANSWER IS:?\s*)?([A-Z])', extracted, re.IGNORECASE)
                if answer_match:
                    extracted = answer_match.group(1).upper()
                
                # Accept any letter in valid_answers set
                if extracted in valid_answers:
                    logger.info(f"Normalized answer: {extracted}")
                    return extracted
                logger.debug(f"Invalid answer format: {extracted} (valid options are {sorted(valid_answers)})")
                return None
                
            except Exception as e:
                logger.error(f"Error in extraction: {str(e)}")
                time.sleep(sleep_time)
        return None

    def verify_answer(self, extracted: str, ground_truth: str, **kwargs) -> bool:
        # Get the options for this specific question
        options = kwargs.get('options')
        if options is None:
            raise ValueError("MMLU-Pro handler requires 'options' in kwargs to determine valid answer range")
            
        num_options = len(options)
        if num_options == 0:
            raise ValueError("MMLU-Pro question has empty options list")
            
        if not extracted:
            logger.debug("Verification failed: extracted answer is None")
            return False
            
        valid_answers = set(chr(i) for i in range(ord('A'), ord('A') + num_options))
            
        # Normalize both to single uppercase letters
        extracted_norm = extracted.strip().upper()
        ground_truth_norm = ground_truth.strip().upper()
        
        # Verify both answers are in valid range
        if extracted_norm not in valid_answers:
            logger.debug(f"Verification failed: extracted answer '{extracted_norm}' not in valid range (A-{chr(ord('A') + num_options - 1)})")
            return False
        if ground_truth_norm not in valid_answers:
            logger.debug(f"Verification failed: ground truth '{ground_truth_norm}' not in valid range (A-{chr(ord('A') + num_options - 1)})")
            return False
        
        logger.debug(f"Verification: {extracted} → {extracted_norm} vs {ground_truth} → {ground_truth_norm}")
        return extracted_norm == ground_truth_norm

def get_dataset_handler(dataset_type: str) -> DatasetHandler:
    """Factory function to get appropriate dataset handler"""
    handlers = {
        "gpqa": GPQAHandler,
        "math": MathHandler,
        "mmlu": MMLUHandler,
        "mmlu_pro": MMLUProHandler
    }
    
    handler_class = handlers.get(dataset_type.lower())
    if not handler_class:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return handler_class()

def evaluate_dataset(
    input_path: str,
    output_path: str,
    dataset_type: str,
    openai_api_key: str,
    parallel: int = 8,
    max_samples: int = None,
    max_rows: int = None
):
    """Main evaluation function"""
    
    # Setup
    logger.add("evaluation.log", rotation="100 MB")
    client = openai.OpenAI(api_key=openai_api_key)
    handler = get_dataset_handler(dataset_type)
    
    # Load dataset
    logger.info(f"Loading {dataset_type} dataset from {input_path}")
    dataset = handler.load_dataset(input_path)
    
    # Truncate sample-related columns if needed
    dataset = handler._truncate_sample_columns(dataset, max_samples)

    # Limit number of rows if specified
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    
    def process_problem(idx):
        """Process a single problem"""
        samples = dataset[idx]['samples'][:max_samples] if max_samples else dataset[idx]['samples']
        extracted_answers = []
        verified_answers = []
        local_extraction_failures = 0
        
        # Add kwargs based on dataset type
        kwargs = {}
        if dataset_type.lower() == 'mmlu_pro':
            kwargs['options'] = dataset[idx]['options']

        for sample in samples:
            try:
                # Extract answer with appropriate kwargs
                answer = handler.extract_answer(dataset[idx]['instruction'], sample, client, **kwargs)
                
                if answer is None:
                    local_extraction_failures += 1
                extracted_answers.append(answer)
                
                # Verify answer with kwargs
                if isinstance(handler, (MathHandler, MMLUProHandler)):
                    is_verified = handler.verify_answer(answer, dataset[idx]['answer'], client=client, **kwargs)
                else:
                    is_verified = handler.verify_answer(answer, dataset[idx]['answer'])
                verified_answers.append(is_verified)
                
            except Exception as e:
                logger.error(f"Error processing sample for problem {idx}: {str(e)}")
                local_extraction_failures += 1
                extracted_answers.append(None)
                verified_answers.append(False)
        
        # Log progress
        correct = sum(verified_answers)
        total = len(verified_answers)
        logger.info(f"Problem {idx}: {correct}/{total} correct ({correct/total:.2%})")
        
        return {
            'idx': idx,
            'extracted_answers': extracted_answers,
            'verified_answers': verified_answers,
            'extraction_failures': local_extraction_failures
        }
    
    # Process problems in parallel
    all_results = []
    extraction_failures = 0
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_idx = {executor.submit(process_problem, idx): idx 
                        for idx in range(len(dataset))}
        
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Processing problems"):
            try:
                result = future.result()
                all_results.append(result)
                extraction_failures += result['extraction_failures']
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error processing problem {idx}: {str(e)}")
                all_results.append({
                    'idx': idx,
                    'extracted_answers': [None] * len(dataset[idx]['samples']),
                    'verified_answers': [False] * len(dataset[idx]['samples']),
                    'extraction_failures': len(dataset[idx]['samples'])
                })
    
    # Sort results by index to maintain alignment
    all_results.sort(key=lambda x: x['idx'])
    
    # Update dataset
    if 'extracted_answers' in dataset.column_names:
        dataset = dataset.remove_columns('extracted_answers')
    dataset = dataset.add_column('extracted_answers', [r['extracted_answers'] for r in all_results])
    
    if 'answer_correct' in dataset.column_names:
        dataset = dataset.remove_columns('answer_correct')
    dataset = dataset.add_column('answer_correct', [r['verified_answers'] for r in all_results])
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    handler.save_dataset(dataset, output_path)
    logger.info(f"\nSaved processed dataset to: {os.path.abspath(output_path)}")
    
    # Calculate and print statistics
    rows_with_correct = sum(1 for r in all_results if any(r['verified_answers']))
    total_correct = sum(sum(r['verified_answers']) for r in all_results)
    total_samples = sum(len(r['verified_answers']) for r in all_results)
    total_rows = len(dataset)
    
    logger.info(f"\nFinal Results:")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total correct: {total_correct}")
    logger.info(f"Sample accuracy: {total_correct/total_samples:.2%}")
    logger.info(f"Rows with correct answer: {rows_with_correct}/{total_rows} ({rows_with_correct/total_rows:.2%})")
    logger.info("\nFailure Analysis:")
    logger.info(f"Extraction failures: {extraction_failures}")

def main():
    parser = argparse.ArgumentParser(description='Unified dataset evaluation script')
    parser.add_argument('--input_path', '-i', required=True, help='Path to input dataset')
    parser.add_argument('--output_path', '-o', required=True, help='Path to save processed dataset')
    parser.add_argument('--dataset_type', '-t', required=True, 
                       choices=['gpqa', 'math', 'mmlu', 'mmlu_pro'],
                       help='Type of dataset to process')
    parser.add_argument('--parallel', '-p', type=int, default=8, 
                       help='Number of parallel workers')
    parser.add_argument('--max_samples', '-m', type=int, default=None,
                       help='Maximum number of samples to process per problem')
    parser.add_argument('--max_rows', '-r', type=int, default=None,
                       help='Maximum number of rows to process')
    
    args = parser.parse_args()
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    evaluate_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        dataset_type=args.dataset_type,
        openai_api_key=openai_api_key,
        parallel=args.parallel,
        max_samples=args.max_samples,
        max_rows=args.max_rows  # Add new parameter
    )

if __name__ == "__main__":
    main()
