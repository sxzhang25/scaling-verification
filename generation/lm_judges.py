from typing import List, Dict, Optional, Union, Tuple
import logging
import torch
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import generate_openai, generate_anthropic, generate_together
import concurrent.futures
from datasets import load_from_disk

def create_evaluation_prompt(instruction: str, sample: str) -> List[Dict[str, str]]:
    """Creates a prompt for evaluating solutions to reasoning tasks including mathematics, 
    science, coding challenges, logic problems and other technical domains."""
    
    system_message = """You are a rigorous evaluator for reasoning tasks spanning mathematics, science, coding, logic, and other technical domains. Your task is to determine if a given solution demonstrates valid reasoning and reaches the correct conclusion.
    
    Evaluate the solution against the relevant criteria such as:
    - Validity of logical arguments and inference steps 
    - Correctness of mathematical calculations and methodology
    - Proper handling of assumptions and edge cases
    - Accuracy of domain-specific concepts and claims
    - Appropriate use of computational techniques (if applicable)
    - Completeness of key reasoning steps
    
    First, analyze the solution by carefully examining the reasoning and steps presented. Provide a detailed analysis that:
    - Identifies the key reasoning components
    - Evaluates the validity of each major step
    - Notes any gaps or potential issues
    - Assesses the correctness of the conclusion
    
    Then respond with EXACTLY "True" if you have high confidence the solution demonstrates valid reasoning and reaches the correct conclusion, or "False" if you have meaningful doubts.
    
    Your response should be structured as:
    1. Your complete analysis and reasoning 
    2. A single line containing only "True" or "False" 

    You must include your final verdict of True or False on a new line at the end of your response."""
    
    user_message = f"""Problem: {instruction}

Solution to evaluate:
{sample}

Analyze the reasoning and solution, then on a new line state True or False:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def create_critique_prompt(instruction: str, sample: str) -> List[Dict[str, str]]:
    """Creates a prompt for paragraph-by-paragraph critique of solutions"""
    
    # Split sample into paragraphs and format with tags
    paragraphs = sample.split('\n\n')
    formatted_solution = '\n'.join(
        f"<paragraph_{i}>\n{para}\n</paragraph_{i}>"
        for i, para in enumerate(paragraphs)
    )
    
    system_message = """You are a rigorous evaluator for mathematical and technical solutions. Your task is to analyze a solution paragraph by paragraph and identify the earliest error, if any exists."""
    
    user_message = f"""The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]
{instruction}

[Solution]
{formatted_solution}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def create_scenegen_evaluation_prompt(instruction: str, sample: str) -> List[Dict[str, str]]:
    """Creates a prompt for evaluating generated scenes."""
    
    system_message = """You are a rigorous evaluator for determining whether a scene is valid and high quality with respect to a given prompt.

    The scene will be represented as a JSON with a list of objects and a floor layout. The floor layout will be represented as a list of vertices and faces.
    Each object will be represented by a unique name, category, position, size, forward direction, right direction, and up direction.
    
    Evaluate the scene against the relevant criteria such as:
    - Whether the right objects are present in the scene
    - Physically plausible placement of objects, such as grounding and intersections
    - Making sure objects are in bounds of the floor layout
    - Accessible and navigable arrangement of the space
    - Proper commonsense relationships between objects and the space

    For instance, if objects intersect or float in the air when they should be on the ground, this is an error. Or if objects are not accessible or
    prevent parts of the room from being accessed, this is an error. If objects do not face the correct direction, this is an error. Check for other 
    errors of similar nature.
    
    Then respond with EXACTLY "True" if you have high confidence the scene is valid and high quality, or "False" if you have meaningful doubts.
    
    Your response should be structured as:
    A single line containing only "True" or "False" 

    You must include your final verdict of True or False on a new line at the end of your response."""
    
    user_message = f"""Prompt: {instruction}

Scene to evaluate:
{sample}

Analyze the scene, then on a new line state True or False:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def create_scenegen_critique_prompt(instruction: str, sample: str) -> List[Dict[str, str]]:
    """Creates a prompt for paragraph-by-paragraph critique of scenes"""
    
    # Split sample into paragraphs and format with tags
    paragraphs = sample.split('\n\n')
    formatted_solution = '\n'.join(
        f"<paragraph_{i}>\n{para}\n</paragraph_{i}>"
        for i, para in enumerate(paragraphs)
    )
    
    system_message = """You are a rigorous evaluator for determining whether a scene is valid and high quality with respect to a given prompt."""
    
    user_message = f"""The following is a prompt and a scene (split into paragraphs, enclosed with tags and indexed from 0):

[Prompt]
{instruction}

[Scene]
{sample}

Your task is to review and critique the different components of the scene. Once you identify an error, return the objects involved in the error and the nature of the error. Otherwise, return "No errors found".

Please put your final answer (i.e., the objects involved in the error and the nature of the error) in \\boxed{{}}."""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

@dataclass
class LMJudgeConfig:
    """Configuration for LM Judge"""
    model_name: str
    provider: Optional[str] = None  # 'vllm', 'openai', or 'anthropic'
    temperature: float = 0.0
    num_verdicts: int = 1
    max_tokens: int = 8192
    batch_size: int = 4
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1

class LMJudge:
    """LM Judge implementation supporting both local vLLM and API inference"""
    
    def __init__(
        self,
        model_name: str,
        provider: Optional[str] = None,
        num_verdicts: int = 1,
        batch_size: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1
    ):
        self.config = LMJudgeConfig(
            model_name=model_name,
            provider=provider,
            num_verdicts=num_verdicts,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size
        )
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        
        # Set up API generator if using API provider
        if provider == 'openai':
            self.generator = generate_openai
        elif provider == 'anthropic':
            self.generator = generate_anthropic
        elif provider == 'together':
            self.generator = generate_together
        
    def __del__(self):
        """Cleanup method to ensure GPU memory is freed"""
        self.unload()
        
    def unload(self):
        """Explicitly unload the model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logging.info(f"Successfully unloaded {self.config.model_name}")
    
    def load(self):
        """Load the model if not already loaded"""
        if self.model is None:
            try:
                if self.config.provider == "vllm":
                    self.model = LLM(
                        model=self.config.model_name,
                        trust_remote_code=True,
                        tensor_parallel_size=self.config.tensor_parallel_size,
                        gpu_memory_utilization=self.config.gpu_memory_utilization,
                        max_num_seqs=self.config.batch_size,
                    )
                else:
                    logging.info(f"Loading LM Judge model {self.config.model_name}")
                    
                    # Configure vLLM parameters
                    self.sampling_params = SamplingParams(
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=0.9
                    )
                    
                    # Initialize vLLM model
                    self.model = LLM(
                        model=self.config.model_name,
                        trust_remote_code=True,
                        tensor_parallel_size=self.config.tensor_parallel_size,
                        gpu_memory_utilization=self.config.gpu_memory_utilization
                    )
                    
                    # Load tokenizer for any preprocessing needs
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True
                    )
                    
                    logging.info("Model loaded successfully")
                    
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                self.unload()  # Ensure cleanup on error
                raise

    def _process_api_batch(
        self,
        batch_instructions: List[str],
        batch_samples: List[str],
        start_idx: int,
        task_type: str
    ) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
        """Process a batch using API calls with parallel execution"""
        scores_by_row = {}
        raw_verdicts_by_row = {}
        #max_workers = 16
        #max_workers = 4  # Default number of parallel API calls
        max_workers = 32
        
        def process_single_sample(args):
            idx, instruction, sample = args
            overall_idx = start_idx + idx
            verdicts = []
            response = None
            
            for _ in range(self.config.num_verdicts):
                if task_type == "scenegen":
                    messages = create_scenegen_evaluation_prompt(instruction, sample)
                else:
                    messages = create_evaluation_prompt(instruction, sample)
                try:
                    response = self.generator(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature
                    )
                    verdict = self._extract_verdict(response)
                    if verdict is not None:
                        verdicts.append(verdict)
                except Exception as e:
                    logging.error(f"API call error for sample {idx}: {e}")
            
            # Average the verdicts (if any were successful)
            score = sum(verdicts) / len(verdicts) if verdicts else None
            return overall_idx, score, response.strip() if score is not None else None
        
        # Create arguments for each sample
        sample_args = [
            (idx, instruction, sample) 
            for idx, (instruction, sample) in enumerate(zip(batch_instructions, batch_samples))
        ]
        
        # Process samples in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_sample, args) for args in sample_args]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    overall_idx, score, raw_verdict = future.result()
                    scores_by_row[overall_idx] = [score]
                    raw_verdicts_by_row[overall_idx] = raw_verdict
                except Exception as e:
                    logging.error(f"Error processing future: {e}")
        
        return scores_by_row, raw_verdicts_by_row

    def _process_vllm_batch(
        self,
        batch_instructions: List[str],
        batch_samples: List[str],
        start_idx: int,
        task_type: str
    ) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
        """Process a batch using vLLM"""
        scores_by_row = {}
        raw_verdicts_by_row = {}
        
        # Create evaluation prompts for batch
        batch_prompts = []
        for instruction, sample in zip(batch_instructions, batch_samples):
            if task_type == "scenegen":
                messages = create_scenegen_evaluation_prompt(instruction, sample)
            else:
                messages = create_evaluation_prompt(instruction, sample)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            batch_prompts.append(prompt)
        
        try:
            outputs = self.model.generate(batch_prompts, self.sampling_params)
            
            # Process each response in the batch
            for idx, output in enumerate(outputs):
                overall_idx = start_idx + idx
                verdicts = []
                
                # For each requested verdict
                for _ in range(self.config.num_verdicts):
                    response = output.outputs[0].text
                    verdict = self._extract_verdict(response)
                    if verdict is not None:
                        verdicts.append(verdict)
                
                # Average the verdicts
                score = sum(verdicts) / len(verdicts) if verdicts else None
                scores_by_row[overall_idx] = [score]
                raw_verdicts_by_row[overall_idx] = response.strip() if score is not None else None
                
        except Exception as e:
            logging.error(f"vLLM processing error: {e}")
            for idx in range(len(batch_instructions)):
                scores_by_row[start_idx + idx] = [None]
                raw_verdicts_by_row[start_idx + idx] = None
        
        return scores_by_row, raw_verdicts_by_row

    def get_scores(
        self,
        instructions: List[str],
        samples: List[str],
        task_type: str,
        batch_size: Optional[int] = None,
    ) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
        """Get scores and raw verdicts for instruction-sample pairs"""
        if len(instructions) != len(samples):
            raise ValueError("Number of instructions and samples must match")
            
        if not self.config.provider and self.model is None:
            self.load()
            
        batch_size = batch_size or self.config.batch_size
        scores_by_row = {}
        raw_verdicts_by_row = {}
        
        for i in tqdm(range(0, len(instructions), batch_size), 
                     desc=f"{self.config.model_name} scoring"):
            batch_instructions = instructions[i:i + batch_size]
            batch_samples = samples[i:i + batch_size]
            
            # Use appropriate processing method
            if self.config.provider in ['openai', 'anthropic', 'together', 'sambanova', 'fireworks']:
                batch_scores, batch_raw_verdicts = self._process_api_batch(
                    batch_instructions, batch_samples, i, task_type
                )
            else:
                batch_scores, batch_raw_verdicts = self._process_vllm_batch(
                    batch_instructions, batch_samples, i, task_type
                )
            
            scores_by_row.update(batch_scores)
            raw_verdicts_by_row.update(batch_raw_verdicts)
        
        return scores_by_row, raw_verdicts_by_row
    
    def _extract_verdict(self, response: str) -> Optional[float]:
        """Extract True/False verdict from response and convert to float"""
        try:
            # Clean and normalize the response
            response = response.strip().lower()
            #print(f"Response: {response}")
            
            # First try to get the last line
            lines = response.split('\n')
            last_line = lines[-1].strip()
            first_line = lines[0].strip()
            
            # Check for explicit TRUE/FALSE
            if 'true' in last_line:
                return 1.0
            elif 'false' in last_line:
                return 0.0
            elif 'true' in first_line:
                return 1.0
            elif 'false' in first_line:
                return 0.0
            
            # Look for verdict in final sentence
            sentences = response.split('.')
            final_sentence = sentences[-1].strip()
            first_sentence = sentences[0].strip() 
            if 'true' in final_sentence:
                return 1.0
            elif 'false' in final_sentence:
                return 0.0
            elif 'true' in first_sentence:
                return 1.0
            elif 'false' in first_sentence:
                return 0.0
                
            return None
            
        except Exception as e:
            logging.error(f"Error extracting verdict: {e}")
            return None
        
    def _extract_critique_verdict(self, response: str) -> Optional[float]:
        """Extract critique verdict from response"""
        try:
            if response is None:
                return None
            
            # Look for the last \boxed{number} pattern in the response
            import re
            matches = list(re.finditer(r'\\boxed{(-?\d+)}', response))
            if matches:
                # Get the last match
                last_match = matches[-1]
                index = int(last_match.group(1))
                
                # Convert to True/False verdict
                # -1 means no errors found (True)
                # Any other index means error found in that paragraph (False)
                return 1.0 if index == -1 else 0.0
            
            logging.warning(f"No \\boxed{{number}} found in response: {response[:100]}...")
            return None
        
        except Exception as e:
            logging.error(f"Error extracting critique verdict: {e}")
            return None

    def get_critique_scores(
        self,
        instructions: List[str],
        samples: List[str],
        task_type: str,
        batch_size: Optional[int] = None,
    ) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
        """Get critique scores and raw verdicts for instruction-sample pairs"""
        if len(instructions) != len(samples):
            raise ValueError("Number of instructions and samples must match")
            
        if not self.config.provider and self.model is None:
            self.load()
            
        batch_size = batch_size or self.config.batch_size
        scores_by_row = {}
        raw_verdicts_by_row = {}
        
        logging.info(f"Processing {len(instructions)} instruction-sample pairs in batches of {batch_size}")
        
        for i in tqdm(range(0, len(instructions), batch_size), 
                      desc=f"{self.config.model_name} critique scoring"):
            batch_instructions = instructions[i:i + batch_size]
            batch_samples = samples[i:i + batch_size]
            
            # Create all prompts for the batch at once
            batch_prompts = []
            for instruction, sample in zip(batch_instructions, batch_samples):
                if task_type == "scenegen":
                    messages = create_scenegen_critique_prompt(instruction, sample)
                else:
                    messages = create_critique_prompt(instruction, sample)
                logging.debug(f"Created critique prompt: {messages}")  # Debug prompt creation
                if self.config.provider not in ['openai', 'anthropic']:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    batch_prompts.append(prompt)
                    logging.debug(f"Applied chat template: {prompt[:200]}...")  # Debug template application
                
            try:
                if self.config.provider in ['openai', 'anthropic']:
                    # Process API models one at a time
                    for idx, (instruction, sample) in enumerate(zip(batch_instructions, batch_samples)):
                        overall_idx = i + idx
                        if task_type == "scenegen":
                            messages = create_scenegen_critique_prompt(instruction, sample)
                        else:
                            messages = create_critique_prompt(instruction, sample)
                        response = self.generator(
                            model=self.config.model_name,
                            messages=messages,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature
                        )
                        verdict = self._extract_critique_verdict(response)
                        scores_by_row[overall_idx] = [verdict] if verdict is not None else [None]
                        raw_verdicts_by_row[overall_idx] = response if verdict is not None else None
                else:
                    # Process vLLM models in batch
                    logging.debug(f"Processing batch of {len(batch_prompts)} prompts with vLLM")
                    outputs = self.model.generate(batch_prompts, self.sampling_params)
                    logging.debug(f"Received {len(outputs)} responses from vLLM")
                    
                    for idx, output in enumerate(outputs):
                        overall_idx = i + idx
                        response = output.outputs[0].text
                        #logging.info(f"Raw response for index {overall_idx}: {response}")  # Log full response
                        
                        verdict = self._extract_critique_verdict(response)
                        #logging.info(f"Extracted verdict for index {overall_idx}: {verdict}")  # Log extracted verdict
                        
                        scores_by_row[overall_idx] = [verdict] if verdict is not None else [None]
                        raw_verdicts_by_row[overall_idx] = response if verdict is not None else None
                        
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
                for idx in range(len(batch_instructions)):
                    overall_idx = i + idx
                    scores_by_row[overall_idx] = [None]
                    raw_verdicts_by_row[overall_idx] = None
        
        logging.info(f"Processed all pairs. Got scores for {sum(1 for s in scores_by_row.values() if s[0] is not None)}/{len(instructions)} samples")
        logging.info(f"Final scores: {scores_by_row}")  # Log final scores
        logging.info(f"Final verdicts: {raw_verdicts_by_row}")  # Log final verdicts
        return scores_by_row, raw_verdicts_by_row

# Dictionary mapping judge names to their model info (path/name and provider)
JUDGE_REGISTRY = {
    # Local vLLM models
    "DeepSeekLlama70B": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "provider": None},
    "DeepSeekQwen32B": {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "provider": None},
    "SkyT1": {"model": "NovaSky-AI/Sky-T1-32B-Preview", "provider": None},
    "Llama-3.3-70B-Instruct": {"model": "meta-llama/Llama-3.3-70B-Instruct", "provider": None},
    "Meta-Llama-3.1-405B-Instruct-quantized.w8a16": {"model": "neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a16", "provider": None},
    "Qwen/Qwen2.5-72B-Instruct": {"model": "Qwen/Qwen2.5-72B-Instruct", "provider": None},
    "QwQ-32B": {"model": "Qwen/QwQ-32B", "provider": None},
    "WizardLM-2-8x22B": {"model": "alpindale/WizardLM-2-8x22B", "provider": None},
    "Mixtral-8x22B-Instruct-v0.1": {"model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "provider": None},

    #########################################################
    # Smaller LM Judges

    "DeepSeekLlama8B": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "provider": None},
    "DeepSeekQwen7B": {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "provider": None},
    "Llama-3.1-8B-Instruct": {"model": "meta-llama/Llama-3.1-8B-Instruct", "provider": None},
    "Gemma-3-12B-Instruct": {"model": "google/gemma-3-12b-it", "provider": None},
    "Gemma-3-4B-Instruct": {"model": "google/gemma-3-4b-it", "provider": None},
    "Phi-4-4B-Instruct": {"model": "microsoft/Phi-4-mini-instruct", "provider": None},
    "Qwen-2.5-7B-Instruct": {"model": "Qwen/Qwen2.5-7B-Instruct", "provider": None},
    "Qwen-2.5-Math-7B-Instruct": {"model": "Qwen/Qwen2.5-Math-7B-Instruct", "provider": None},
    "Mistral-7B-Instruct-v0.2": {"model": "mistralai/Mistral-7B-Instruct-v0.2", "provider": None},

    #########################################################
    
    # API models
    "Llama-3.1-8B-Instruct-Together": {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "provider": "together"},
    "Llama-3.1-70B-Instruct-Together": {"model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "provider": "together"},
    "GPT-4o": {"model": "gpt-4o", "provider": "openai"},
    "GPT-4o-mini": {"model": "gpt-4o-mini", "provider": "openai"},
    "Claude-3-7-Sonnet": {"model": "claude-3-7-sonnet-latest", "provider": "anthropic"},
    "Claude-3-5-Sonnet": {"model": "claude-3-5-sonnet-latest", "provider": "anthropic"},
    "Claude-3-5-Haiku": {"model": "claude-3-5-haiku-latest", "provider": "anthropic"},
}

def get_judge(
    judge_name: str,
    num_verdicts: int = 1,
    batch_size: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1
) -> LMJudge:
    """Factory function to get the appropriate judge instance"""
    if judge_name not in JUDGE_REGISTRY:
        raise ValueError(f"Unknown judge: {judge_name}. Available judges: {list(JUDGE_REGISTRY.keys())}")
    
    model_info = JUDGE_REGISTRY[judge_name]
    judge = LMJudge(
        model_name=model_info["model"],
        provider=model_info["provider"],
        num_verdicts=num_verdicts,
        batch_size=batch_size,
        temperature=temperature,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size
    )
    
    return judge
