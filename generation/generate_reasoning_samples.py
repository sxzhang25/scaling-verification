import argparse
from typing import List, Dict, Optional
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import logging
from tqdm import tqdm
import time
from queue import Queue, Empty
from threading import Thread

from utils import generate_openai, generate_anthropic, generate_together

# Add model provider mapping
MODEL_PROVIDERS = {
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
    "o3-mini-2025-01-31": "openai",
    "o1-2024-12-17": "openai",
    "claude-3-5-sonnet-latest": "anthropic",
    "claude-3-5-haiku-latest": "anthropic",
    "deepseek-ai/DeepSeek-R1": "together",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "together",
}

# Add provider to generator mapping
PROVIDER_GENERATORS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "together": generate_together,
}

COLLEGE_SUBJECTS = {
    'college_biology', 'college_chemistry', 'college_computer_science', 
    'college_mathematics', 'college_medicine', 'college_physics'
}

PROFESSIONAL_SUBJECTS = {
    'professional_accounting', 'professional_law', 'professional_medicine', 
    'professional_psychology', 
}

def load_mmlu_pro_dataset(split: str = "test", seed: int = 42) -> Dataset:
    """
    Load and prepare MMLU-Pro dataset.
    
    Args:
        split: Dataset split to use ('test' or 'validation')
        seed: Random seed for shuffling
    
    Returns:
        Dataset formatted for reasoning generation
    """
    # Load dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    
    # Transform into required format
    def format_problem(example):
        # Create problem text combining question and choices with letter labels
        choices_text = "\n".join(f"{chr(65 + idx)}) {choice}" 
                               for idx, choice in enumerate(example['options']))
        problem = (
            f"Question: {example['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Please solve this step by step, then output your answer on a new line as 'The answer is: X' "
            f"where X is the letter corresponding to your choice."
        )
        
        return {
            'problem': problem,
            'answer': chr(65 + example['answer_index']),  # Convert index to letter
            'subject': example['category']
        }
    
    transformed_ds = ds.map(format_problem)
    
    # Shuffle the dataset with a fixed seed
    transformed_ds = transformed_ds.shuffle(seed=seed)
    
    # Print an example prompt
    print("\nExample MMLU-Pro prompt:")
    print("-" * 80)
    print(transformed_ds[0]['problem'])
    print("-" * 80)
    print(f"Correct answer: {transformed_ds[0]['answer']}")
    
    return transformed_ds

def load_mmlu_dataset(subjects: Optional[List[str]] = None, split: str = "test", category: str = "college") -> Dataset:
    """
    Load and prepare MMLU dataset.
    
    Args:
        subjects: List of subject names to include. If None, uses all college subjects.
        split: Dataset split to use ('test', 'validation', or 'dev')
        category: Dataset category ('college' or 'professional')
    
    Returns:
        Dataset formatted for reasoning generation
    """
    if subjects is None:
        subjects = list(COLLEGE_SUBJECTS if category == "college" else PROFESSIONAL_SUBJECTS)
    
    # Load and combine all requested subjects
    all_data = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject)[split]
            all_data.extend([{**item} for item in ds])
        except Exception as e:
            logging.warning(f"Failed to load subject {subject}: {e}")
            continue
    
    # Convert to Dataset format
    ds = Dataset.from_list(all_data)
    
    # Transform into required format
    def format_problem(example):
        # Create problem text combining question and choices with letter labels
        choices_text = "\n".join(f"{chr(65 + idx)}) {choice}" for idx, choice in enumerate(example['choices']))
        problem = (
            f"Question: {example['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Please solve this step by step, then output your answer on a new line as 'The answer is: X' "
            f"where X is A, B, C, or D."
        )
        
        return {
            'problem': problem,
            'answer': chr(65 + example['answer']),  # Store correct answer as letter
            'subject': example['subject']
        }
    
    transformed_ds = ds.map(format_problem)
    
    # Print an example prompt
    print("\nExample MMLU prompt:")
    print("-" * 80)
    print(transformed_ds[0]['problem'])
    print("-" * 80)
    print(f"Correct answer: {transformed_ds[0]['answer']}")
    
    return transformed_ds

def load_math_dataset(max_rows: Optional[int] = None) -> Dataset:
    """
    Load and prepare MATH dataset for reasoning generation.
    
    Args:
        max_rows: Maximum number of rows to include
    
    Returns:
        Dataset formatted for reasoning generation
    """
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    
    # Get the first available split (either 'train' or 'test')
    split_name = 'train' if 'train' in dataset else 'test'
    dataset = dataset[split_name]
    
    # Limit rows if specified
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    
    # Transform into required format
    def format_problem(example):
        problem = (
            f"Problem: {example['problem']}\n\n"
            f"Please solve this step by step, then output your answer on a new line as 'The answer is: X'"
        )
        
        return {
            'problem': problem,
            'answer': example['solution'],  # Keep original solution
            'subject': example.get('subject', 'mathematics')  # Add subject if not present
        }
    
    transformed_ds = dataset.map(format_problem)
    
    # Print an example prompt
    print("\nExample MATH prompt:")
    print("-" * 80)
    print(transformed_ds[0]['problem'])
    print("-" * 80)
    print(f"Expected answer format: The answer is: {transformed_ds[0]['answer']}")
    
    return transformed_ds


def load_gpqa_dataset(config_name: str) -> Dataset:
    """
    Load and prepare GPQA dataset from the new source.
    
    Args:
        config_name: The HuggingFace config name to load (e.g., "gpqa_main" or "gpqa_diamond")
    
    Returns:
        Dataset formatted for reasoning generation wrapped in "data" dict
    """
    # Load dataset with the specified config
    ds = load_dataset("Idavidrein/gpqa", config_name)['train']
    
    # Determine the type from config name
    dataset_type = "diamond" if "diamond" in config_name else "main"
    
    def format_problem(example):
        # Create options array
        options = [
            example['Correct Answer'],
            example['Incorrect Answer 1'], 
            example['Incorrect Answer 2'],
            example['Incorrect Answer 3']
        ]
        
        # Randomize the order but keep track of correct answer position
        import random
        correct_answer = example['Correct Answer']
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        correct_letter = chr(65 + correct_idx)  # Convert to A, B, C, D
        
        # Format choices with letters for the problem text
        choices_text = "\n".join(f"{chr(65 + idx)}) {choice}" 
                                for idx, choice in enumerate(options))
        
        # Create the formatted problem
        problem = (
            f"Question: {example['Question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Please solve this step by step, then output your answer on a new line as 'The answer is: X' "
            f"where X is the letter corresponding to your choice."
        )
        
        return {
            'problem': problem,
            'original_instruction': example['Question'],
            'type': dataset_type,
            'options': options,
            'answer': correct_letter,
            'correct_answer': correct_letter,
            'subject': 'science'
        }
    
    transformed_ds = ds.map(format_problem, remove_columns=ds.column_names)
    
    # Print an example prompt
    print(f"\nExample GPQA {config_name} prompt:")
    print("-" * 80)
    print(transformed_ds[0]['problem'])
    print("-" * 80)
    print(f"Correct answer: {transformed_ds[0]['answer']}")
    print(f"Type: {transformed_ds[0]['type']}")
    print(f"Subject: {transformed_ds[0]['subject']}")
    print(f"Number of examples: {len(transformed_ds)}")
    
    return transformed_ds


class ReasoningGenerator:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_length: int = 2048,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        chat_format: Optional[str] = None,
        samples_per_instruction: int = 1,
        num_workers: int = 1,
    ):
        self.model_name = model_name
        self.config = {
            "batch_size": batch_size,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "samples_per_instruction": samples_per_instruction,
            "gpu_memory_utilization": gpu_memory_utilization,
            "num_workers": num_workers
        }
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup GPU parallelism
        self.gpu_count = torch.cuda.device_count()
        self.tensor_parallel_size = tensor_parallel_size or min(8, self.gpu_count)
        
        # For S1 model, always use chat format
        if "s1" in model_name.lower():
            self.chat_format = "s1"
            self.logger.info("Detected S1 model, enabling chat format")
        else:
            self.chat_format = chat_format

        # Determine if we're using an API provider or vLLM
        self.provider = MODEL_PROVIDERS.get(model_name)
        if self.provider:
            self.generator = PROVIDER_GENERATORS[self.provider]
            self.logger.info(f"Using {self.provider} API for model {model_name}")
        else:
            self._setup_model(model_name)
            self.logger.info(f"Using vLLM for model {model_name}")

    def _setup_model(self, model_name: str):
        """Initialize model and tokenizer"""
        self.logger.info(f"Loading model: {model_name}")
        
        # Initialize vLLM model
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.config["max_length"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            trust_remote_code=True
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            trust_remote_code=True
        )
            
        # Get stop token ids
        if self.chat_format == "s1":
            # Get S1-specific stop tokens
            s1_stop_tokens = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
            if not s1_stop_tokens:
                raise ValueError("Failed to get valid stop tokens for S1 format")
                
            # Combine with default tokenizer stop tokens
            self.stop_token_ids = s1_stop_tokens + (
                [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            )
            assert self.tokenizer.eos_token_id is not None, "EOS token ID is not set"
            self.logger.info(f"Using S1 chat format with combined stop tokens: {self.stop_token_ids}")
        else:
            # Use default tokenizer stop tokens
            self.stop_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            assert self.tokenizer.eos_token_id is not None, "EOS token ID is not set"
            self.logger.info(f"Using default stop tokens: {self.stop_token_ids}")
            
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.config["max_new_tokens"],
            min_tokens=0,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            stop_token_ids=self.stop_token_ids
        )

    def _process_api_batch(self, queries: List[str], batch_idx: int) -> List[List[str]]:
        """Process a batch of queries using API providers"""
        # For API providers, we process all samples for a single query at once
        query = queries[0]  # All queries in the batch are the same
        samples = []
        
        # Generate all samples for this query
        for _ in range(len(queries)):  # len(queries) equals samples_per_instruction
            messages = [{"role": "user", "content": query}]
            if self.chat_format == "s1":
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": f"{query}\nAt the end of your response, add your answer on a newline with the format \"The answer is: <answer here>\""}
                ]
            
            for attempt in range(5):
                try:
                    response = self.generator(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.config["max_new_tokens"],
                        temperature=self.config["temperature"],
                    )
                    samples.append(response)
                    break
                except Exception as e:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
            else:
                self.logger.error(f"Failed to process query after 5 attempts")
                samples.append(None)
        
        return [samples]  # Return all samples for this query as a single batch result

    def _process_vllm_batch(self, batch: List[str], batch_idx: int) -> List[str]:
        """Process a batch using vLLM"""
        try:
            # Format prompts if using chat format
            if self.chat_format:
                batch = [self._format_prompt(query) for query in batch]
                
            # Generate with vLLM
            outputs = self.model.generate(batch, sampling_params=self.sampling_params)
            
            # Extract generated text
            decoded = [output.outputs[0].text for output in outputs]
            
            return decoded
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            return ["ERROR: " + str(e)] * len(batch)

    def _process_batch(self, batch: List[str], batch_idx: int) -> List[str]:
        """Process a single batch of queries"""
        if self.provider:
            return self._process_api_batch(batch, batch_idx)
        else:
            return self._process_vllm_batch(batch, batch_idx)

    def _process_batches_thread(self, problem_queue: Queue, results: Dict[int, List[str]]):
        """Process batches in a thread"""
        while True:
            try:
                idx, problem, current_batch = problem_queue.get_nowait()
                batch_results = self._process_batch(current_batch, idx)
                results[idx] = batch_results
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
                results[idx] = [None] * len(current_batch)

    def _format_prompt(self, query: str) -> str:
        """Format the prompt according to the chat template"""
        if self.chat_format == "s1":
            return (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n" 
                f"{query}\nAt the end of your response, add your answer on a newline with the format \"The answer is: <answer here>\"\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return query + "\nAt the end of your response, add your answer on a newline with the format \"The answer is: <answer here>\""

    def generate_samples(self, dataset: Dataset, output_file: str, max_rows: Optional[int] = None):
        """Generate reasoning samples from dataset with parallel processing"""
        # Load dataset
        self.logger.info(f"Processing dataset with {len(dataset)} rows")
        
        # If dataset is a DatasetDict, get the appropriate split
        if hasattr(dataset, 'keys'):
            # Try to get test split first, then train if test doesn't exist
            if 'test' in dataset:
                dataset = dataset['test']
            elif 'train' in dataset:
                dataset = dataset['train']
            else:
                raise ValueError(f"Dataset must have either 'test' or 'train' split. Available splits: {list(dataset.keys())}")
        
        # Limit rows if specified
        if max_rows is not None:
            self.logger.info(f"Limiting to {max_rows} rows")
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        
        # Remove specified columns if they exist
        columns_to_remove = ['samples', 'answer_correct']
        dataset = dataset.remove_columns([col for col in columns_to_remove if col in dataset.column_names])
        
        # If 'instruction' exists in dataset, rename it to 'original_instruction'
        if 'instruction' in dataset.column_names:
            dataset = dataset.rename_column('instruction', 'original_instruction')
        
        self.logger.info(f"Processing {len(dataset)} problems, {self.config['samples_per_instruction']} samples each")
        
        all_samples = []
        problem_queue = Queue()
        results = {}

        # Prepare batches and add to queue - UPDATED LOGIC
        if self.provider:
            # For API providers, create one batch per problem with all samples
            for idx in range(len(dataset)):
                problem = dataset[idx]["problem"]
                current_batch = [problem] * self.config["samples_per_instruction"]
                problem_queue.put((idx, problem, current_batch))
        else:
            # For vLLM, keep original batching logic
            for idx in range(len(dataset)):
                problem = dataset[idx]["problem"]
                samples_remaining = self.config["samples_per_instruction"]
                while samples_remaining > 0:
                    current_batch_size = min(self.config["batch_size"], samples_remaining)
                    current_batch = [problem] * current_batch_size
                    problem_queue.put((idx, problem, current_batch))
                    samples_remaining -= current_batch_size

        # Create and start threads if using API, otherwise process directly
        if self.provider:
            threads = []
            for _ in range(min(self.config["num_workers"], problem_queue.qsize())):
                thread = Thread(target=self._process_batches_thread, args=(problem_queue, results))
                thread.start()
                threads.append(thread)

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            # Process using single thread for vLLM
            self._process_batches_thread(problem_queue, results)

        # Process results in order
        for idx in tqdm(range(len(dataset)), desc="Processing results"):
            row_data = dict(dataset[idx])
            row_data["instruction"] = row_data["problem"]
            # Flatten all samples for this index into a single list
            all_result_samples = []
            for batch_results in results.get(idx, []):
                if isinstance(batch_results, list):
                    all_result_samples.extend(batch_results)
                else:
                    all_result_samples.append(batch_results)
            row_data["samples"] = all_result_samples[:self.config["samples_per_instruction"]]
            all_samples.append(row_data)

        # Convert to Dataset and save
        output_dataset = Dataset.from_list(all_samples)
        output_dataset.save_to_disk(output_file)
        
        self.logger.info(f"Generation complete. Results saved to {output_file}")
        return output_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning samples from dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, choices=["mmlu", "mmlu_pro", "math", "gpqa", "gpqa_diamond"], help="Benchmark dataset name")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for dataset")
    parser.add_argument("--max_rows", type=int, help="Maximum number of rows to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--tensor_parallel_size", type=int, help="Tensor parallel size for vLLM (default: auto)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="GPU memory utilization (0.0 to 1.0)")
    parser.add_argument("--chat_format", type=str, choices=["s1"], help="Chat format to use (auto-detected for S1 models)")
    parser.add_argument("--samples_per_instruction", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--mmlu_subjects", nargs='+', help="Specific MMLU subjects to include (default: all subjects from selected category)")
    parser.add_argument("--mmlu_category", type=str, choices=['college', 'professional'], default='college', help="Category of MMLU subjects to use (default: college)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads for API calls")
    
    args = parser.parse_args()
    
    generator = ReasoningGenerator(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        chat_format=args.chat_format,
        samples_per_instruction=args.samples_per_instruction,
        num_workers=args.num_workers,
    )
    
    # Handle dataset loading
    if args.dataset.lower() == "mmlu":
        dataset = load_mmlu_dataset(subjects=args.mmlu_subjects, category=args.mmlu_category)
    elif args.dataset.lower() == "mmlu_pro":
        dataset = load_mmlu_pro_dataset()
    elif args.dataset.lower() == "gpqa":
        dataset = load_gpqa_dataset("gpqa_main")
    elif args.dataset.lower() == "gpqa_diamond":
        dataset = load_gpqa_dataset("gpqa_diamond")
    elif args.dataset.lower() == "math":
        dataset = load_math_dataset(args.max_rows)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    generator.generate_samples(dataset=dataset, output_file=args.output_path, max_rows=args.max_rows)

if __name__ == "__main__":
    main()