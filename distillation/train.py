import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel
from typing import List, Dict, Optional, Union, Tuple
import logging
import os
import wandb
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training the cross-encoder model."""
    # Model parameters
    model_name: str  # Either "Alibaba-NLP/gte-Qwen2-1.5B-instruct" or "answerdotai/ModernBERT-large"
    max_length: int = 8192
    mlp_hidden_dims: List[int] = None  # If None, will be set based on model type
    model_size: str = "70B"  # Either "70B" or "8B" for dataset model size
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    checkpoint_frequency: Optional[int] = None  # Save checkpoint every N steps
    
    # Memory optimization
    gradient_accumulation_steps: int = 4  # Accumulate gradients over multiple steps
    max_grad_norm: float = 1.0  # Gradient clipping
    use_amp: bool = True  # Use automatic mixed precision
    
    # Data parameters
    dataset_path: str = None
    all_datasets: bool = False  # Flag to use all three hardcoded datasets
    score_columns: List[str] = None
    use_naive_ensemble: bool = False
    max_rows: Optional[int] = None
    start_row: Optional[int] = None  # Starting row index for training
    end_row: Optional[int] = None    # Ending row index for training (exclusive)
    shuffle_samples: bool = False    # Whether to shuffle samples within each row
    
    # Output settings
    output_dir: str = "checkpoints"
    use_wandb: bool = True
    
    # Early stopping parameters
    early_stopping_patience: int = 3  # Number of epochs to wait before early stopping
    early_stopping_min_delta: float = 0.00000001  # Minimum change in validation loss to be considered as improvement
    
    # Other settings
    seed: int = 42
    
    def __post_init__(self):
        """Set default MLP dimensions based on model type."""
        if self.mlp_hidden_dims is None:
            if "Qwen2" in self.model_name:
                if "7B" in self.model_name:
                    self.mlp_hidden_dims = [3584, 1792, 896]  # For Qwen2-7B (3584-dim embeddings)
                else:
                    self.mlp_hidden_dims = [1536, 768, 384]  # For other Qwen2 models (1536-dim embeddings)
            else:
                self.mlp_hidden_dims = [1024, 512, 256]  # For ModernBERT (1024-dim embeddings)
        
        # Validate dataset parameters
        if self.dataset_path is None and not self.all_datasets:
            raise ValueError("Either dataset_path or all_datasets must be specified")
        if self.dataset_path is not None and self.all_datasets:
            raise ValueError("Cannot specify both dataset_path and all_datasets")

class MLPHead(nn.Module):
    """MLP head for regression task."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer for regression
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class CustomCrossEncoder(nn.Module):
    """Custom cross-encoder model combining a base model with an MLP head."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Initialize base model and tokenizer
        if "Qwen2" in config.model_name:
            # Load the base model directly
            self.base_model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                use_cache=False,  # Disable KV cache
                torch_dtype=torch.bfloat16,  # Use BF16 instead of FP16
                device_map="auto"  # Enable automatic device mapping
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
            # Set embedding dimension based on model size
            if "7B" in config.model_name:
                self.embedding_dim = 3584  # For Qwen2-7B
            else:
                self.embedding_dim = 1536  # For other Qwen2 models
        else:
            # For ModernBERT and other models
            model_kwargs = {
                "torch_dtype": torch.bfloat16,  # Use BF16 instead of FP16
                "device_map": "auto",  # Enable automatic device mapping
                # "use_flash_attention_2": True,  # Enable flash attention
                # "attn_implementation": "eager",  # Use eager attention implementation
                "output_hidden_states": True  # Ensure we get hidden states
            }
            # Use the base model directly instead of the masked LM version
            self.base_model = AutoModel.from_pretrained(
                config.model_name,
                **model_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.embedding_dim = 1024
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize MLP head in BF16
        self.mlp_head = MLPHead(
            input_dim=self.embedding_dim,
            hidden_dims=config.mlp_hidden_dims,
            dropout_rate=config.dropout_rate
        ).to(torch.bfloat16)  # Keep MLP head in BF16
        
        # Enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()
        
        # Disable memory efficient attention
        if hasattr(self.base_model, "config"):
            self.base_model.config.use_memory_efficient_attention = False

    def load_finetuned_checkpoint(self, checkpoint_path: str):
        """Load a finetuned checkpoint saved during training.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt) saved during training,
                           or a HuggingFace model repository ID
        """
        logger.info(f"Loading finetuned checkpoint from: {checkpoint_path}")
        
        # Check if this is a HuggingFace repo
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            logger.info("Loading from HuggingFace repository...")
            try:
                # First load the base model from the original model name
                base_model_name = self.config.model_name
                logger.info(f"Loading base model from: {base_model_name}")
                self.base_model = AutoModel.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    output_hidden_states=True
                )
                # Move model to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.base_model = self.base_model.to(device)
                logger.info(f"Successfully loaded base model from {base_model_name} to {device}")
                
                # Now load the checkpoint model to get the weights
                logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
                model_path = hf_hub_download(repo_id=checkpoint_path, filename="pytorch_model.bin")
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Filter out base model weights and load them
                base_model_state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() 
                                       if k.startswith('base_model.')}
                if base_model_state_dict:
                    self.base_model.load_state_dict(base_model_state_dict)
                    logger.info("Successfully loaded base model weights from checkpoint")
                
                # Filter out MLP head weights and load them
                mlp_state_dict = {k.replace('mlp_head.', ''): v for k, v in state_dict.items() 
                                if k.startswith('mlp_head.')}
                if mlp_state_dict:
                    # Move MLP head to the same device as base model
                    self.mlp_head = self.mlp_head.to(device)
                    self.mlp_head.load_state_dict(mlp_state_dict)
                    logger.info("Successfully loaded MLP head weights from checkpoint")
                
                # Set to evaluation mode
                self.eval()
                
                return self
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace: {e}")
                raise
        
        # If not a HF repo or loading failed, try loading local file
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle both direct state dict and wrapped state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Split state dict into base model and MLP head
            base_model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('mlp_head.')}
            mlp_state_dict = {k.replace('mlp_head.', ''): v for k, v in state_dict.items() if k.startswith('mlp_head.')}
            
            # Load base model state dict
            if base_model_state_dict:
                self.base_model.load_state_dict(base_model_state_dict)
                logger.info("Successfully loaded base model weights")
            
            # Load MLP head state dict
            if mlp_state_dict:
                # Move MLP head to the same device as base model
                device = next(self.base_model.parameters()).device
                self.mlp_head = self.mlp_head.to(device)
                self.mlp_head.load_state_dict(mlp_state_dict)
                logger.info("Successfully loaded MLP head weights")
            
            # Set to evaluation mode
            self.eval()
            
            return self
        except Exception as e:
            logger.error(f"Failed to load local checkpoint: {e}")
            raise

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # For ModernBERT, we need to use the last hidden state from the model's layers
        if "ModernBERT" in self.config.model_name:
            # Get the last hidden state from the model's layers
            last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
            # Use the CLS token (first token) embedding
            cls_embedding = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        else:
            # For other models (like Qwen2)
            last_hidden_state = outputs.hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :]  # Keep in BF16
        
        # Pass through MLP head
        output = self.mlp_head(cls_embedding)
        return output.squeeze(-1)  # Remove the last dimension to match label shape

class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder training."""
    
    def __init__(self, samples: List[Dict], model: CustomCrossEncoder):
        self.samples = samples
        self.model = model
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize the input
        encoded = self.model.tokenizer(
            text=sample["texts"][0],
            text_pair=sample["texts"][1],
            truncation=True,
            max_length=self.model.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": torch.tensor(sample["label"], dtype=torch.float)
        }

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_data(config: TrainingConfig) -> Tuple[List[Dict], List[Dict]]:
        """Load and process the dataset(s)."""
        from datasets import load_dataset
        
        all_samples = []  # List of all instruction-sample pairs
        
        # Determine which datasets to load
        if config.all_datasets:
            model_size = config.model_size
            dataset_paths = [
                f"hazyresearch/MATH500_with_Llama_3.1_{model_size}_Instruct_v1",
                f"hazyresearch/GPQA_with_Llama_3.1_{model_size}_Instruct_v1",
                f"hazyresearch/MMLU-Pro_with_Llama_3.1_{model_size}_Instruct_v1"
            ]
        else:
            dataset_paths = [config.dataset_path]
        
        for dataset_path in dataset_paths:
            logger.info(f"Loading dataset: {dataset_path}")
            dataset = load_dataset(dataset_path)
            data = dataset["data"]
            
            # Calculate row range
            start_idx = config.start_row if config.start_row is not None else 0
            end_idx = config.end_row if config.end_row is not None else len(data)
            if config.max_rows is not None:
                end_idx = min(end_idx, start_idx + config.max_rows)
            
            logger.info(f"Processing rows {start_idx} to {end_idx} from {dataset_path}...")
            
            instructions = data["instruction"][start_idx:end_idx]
            all_samples_list = data["samples"][start_idx:end_idx]
            
            # Process scores
            if config.use_naive_ensemble:
                score_columns = [col for col in data.column_names if col.endswith(("_verdicts", "_scores"))]
                logger.info(f"Using naive ensemble with {len(score_columns)} columns:")
                for i, col in enumerate(score_columns, 1):
                    logger.info(f"{i}. {col}")
            else:
                score_columns = config.score_columns
                if not score_columns:
                    raise ValueError("Either score_columns or use_naive_ensemble must be specified")
            
            score_arrays = {
                col: np.array([scores for scores in data[col][start_idx:end_idx]])
                for col in score_columns
            }
            
            # Create samples grouped by instruction
            for idx in tqdm(range(end_idx - start_idx)):
                instruction = instructions[idx]
                samples = all_samples_list[idx]
                
                # Calculate scores
                scores = []
                for column in score_columns:
                    column_scores = score_arrays[column][idx]
                    # Check if score list length matches samples length
                    if len(column_scores) != len(samples):
                        raise ValueError(
                            f"Score list length mismatch for {column} at index {idx}. "
                            f"Expected {len(samples)} scores for {len(samples)} samples, "
                            f"but got {len(column_scores)} scores."
                        )
                    scores.append(column_scores)
                
                # Average scores across all columns
                final_scores = np.mean(scores, axis=0)
                
                # Create samples for this instruction
                instruction_samples = [
                    {
                        "texts": [instruction, samples[i]],
                        "label": float(score)
                    }
                    for i, score in enumerate(final_scores)
                ]
                
                # Shuffle samples within this instruction if requested
                if config.shuffle_samples:
                    np.random.shuffle(instruction_samples)
                
                all_samples.extend(instruction_samples)
        
        # Shuffle all samples
        np.random.shuffle(all_samples)
        
        # Split into train/val with 90/10 ratio
        train_size = int(0.9 * len(all_samples))
        train_samples = all_samples[:train_size]
        eval_samples = all_samples[train_size:]
        
        logger.info(f"Created {len(train_samples)} training and {len(eval_samples)} evaluation samples")
        return train_samples, eval_samples

class CrossEncoderTrainer:
    """Handles training and evaluation of the cross-encoder model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Set up output directories
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"training_{self.run_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)
        
        # Initialize model
        self.model = CustomCrossEncoder(config).to(self.device)
        #breakpoint()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="cross-encoder-training",
                name=self.run_name,
                config=vars(config),
                settings=wandb.Settings(init_timeout=180)  # Increase timeout to 5 minutes
            )
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_path = os.path.join(
            self.config.output_dir,
            f"best_model_{self.run_name}.pt"
        )
    
    def evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_eval_loss = 0
        num_eval_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device, dtype=torch.bfloat16)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                    if outputs.dim() > labels.dim():
                        outputs = outputs.squeeze()
                    if labels.dim() > outputs.dim():
                        labels = labels.squeeze()
                    loss = nn.MSELoss()(outputs, labels)
                total_eval_loss += loss.item()
                num_eval_batches += 1
        
        return total_eval_loss / num_eval_batches
    
    def train(self):
        """Train the model."""
        # Log hyperparameters
        logger.info("-" * 50)
        logger.info("Training Hyperparameters:")
        logger.info("-" * 50)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Max Length: {self.config.max_length}")
        logger.info(f"MLP Hidden Dimensions: {self.config.mlp_hidden_dims}")
        logger.info(f"Number of Epochs: {self.config.num_epochs}")
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info(f"Learning Rate: {self.config.learning_rate}")
        logger.info(f"Warmup Ratio: {self.config.warmup_ratio}")
        logger.info(f"Weight Decay: {self.config.weight_decay}")
        logger.info(f"Dropout Rate: {self.config.dropout_rate}")
        logger.info(f"Gradient Accumulation Steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Max Gradient Norm: {self.config.max_grad_norm}")
        logger.info(f"Use AMP: {self.config.use_amp}")
        logger.info(f"Dataset Path: {self.config.dataset_path}")
        logger.info(f"All Datasets: {self.config.all_datasets}")
        logger.info(f"Score Columns: {self.config.score_columns}")
        logger.info(f"Use Naive Ensemble: {self.config.use_naive_ensemble}")
        logger.info(f"Max Rows: {self.config.max_rows}")
        logger.info(f"Output Directory: {self.config.output_dir}")
        logger.info(f"Use Wandb: {self.config.use_wandb}")
        logger.info(f"Early Stopping Patience: {self.config.early_stopping_patience}")
        logger.info(f"Early Stopping Min Delta: {self.config.early_stopping_min_delta}")
        logger.info(f"Random Seed: {self.config.seed}")
        logger.info("-" * 50)
        
        # Load data
        train_samples, eval_samples = DataProcessor.load_data(self.config)
        
        # Create datasets
        train_dataset = CrossEncoderDataset(train_samples, self.model)
        eval_dataset = CrossEncoderDataset(eval_samples, self.model)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training loop
        logger.info("Starting training...")
        total_steps = 0
        running_train_loss = 0
        batches_since_last_log = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device, dtype=torch.bfloat16)
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                    if outputs.dim() > labels.dim():
                        outputs = outputs.squeeze()
                    if labels.dim() > outputs.dim():
                        labels = labels.squeeze()
                    loss = nn.MSELoss()(outputs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Accumulate loss
                running_train_loss += loss.item() * self.config.gradient_accumulation_steps
                batches_since_last_log += 1
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_steps += 1
                    
                    # Log training loss every 10 steps
                    if total_steps % 10 == 0:
                        avg_train_loss = running_train_loss / batches_since_last_log
                        logger.info(f"Step {total_steps} (processed {batches_since_last_log} batches): Training Loss: {avg_train_loss:.4f}")
                        if self.config.use_wandb:
                            wandb.log({
                                "train_loss": avg_train_loss,
                                "step": total_steps,
                                "batches_processed": batches_since_last_log
                            })
                        running_train_loss = 0
                        batches_since_last_log = 0
                        
                        # Run validation every 200 steps
                        if total_steps % 1000000 == 0:
                            logger.info(f"Running validation at step {total_steps} (processed {total_steps * self.config.gradient_accumulation_steps} batches)...")
                            
                            # Clear memory before validation
                            torch.cuda.empty_cache()
                            if hasattr(self.model, 'base_model'):
                                self.model.base_model.zero_grad(set_to_none=True)
                            
                            val_loss = self.evaluate(eval_loader)
                            logger.info(f"Step {total_steps}: Validation Loss: {val_loss:.4f}")
                            
                            if self.config.use_wandb:
                                wandb.log({
                                    "val_loss": val_loss,
                                    "step": total_steps,
                                    "epoch": epoch + 1
                                })
                            
                            # Save checkpoint if checkpoint_frequency is specified
                            if self.config.checkpoint_frequency is not None and total_steps % self.config.checkpoint_frequency == 0:
                                checkpoint_path = os.path.join(
                                    self.config.output_dir,
                                    f"checkpoint_step{total_steps}_{self.run_name}.pt"
                                )
                                torch.save(self.model.state_dict(), checkpoint_path)
                                logger.info(f"Saved checkpoint at step {total_steps} to {checkpoint_path}")
                            
                            # Clear memory after validation
                            torch.cuda.empty_cache()
                            if hasattr(self.model, 'base_model'):
                                self.model.base_model.zero_grad(set_to_none=True)
                            
                            # Reset model to training mode
                            self.model.train()
                
                total_train_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Log final training loss for the epoch
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Final Training Loss: {avg_train_loss:.4f}")
            
            # Run validation at the end of each epoch
            # Clear memory before validation
            torch.cuda.empty_cache()
            if hasattr(self.model, 'base_model'):
                self.model.base_model.zero_grad(set_to_none=True)
            
            val_loss = self.evaluate(eval_loader)
            logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch + 1
                })
            
            # Check for improvement and save best model
            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Clear memory after validation
            torch.cuda.empty_cache()
            if hasattr(self.model, 'base_model'):
                self.model.base_model.zero_grad(set_to_none=True)
            
            # Reset model to training mode
            self.model.train()
            
            # Early stopping check
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info(f"Training completed. Best model saved to: {self.best_model_path}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.config.use_wandb:
            wandb.finish()

def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a custom cross-encoder model")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True,
                      help="Base model to use (Alibaba-NLP/gte-Qwen2-1.5B-instruct or answerdotai/ModernBERT-large)")
    parser.add_argument("--max_length", type=int, default=8192,
                      help="Maximum sequence length")
    parser.add_argument("--mlp_hidden_dims", type=str, default=None,
                      help="Comma-separated list of MLP hidden dimensions")
    parser.add_argument("--model_size", type=str, choices=["70B", "8B"], default="70B",
                      help="Model size for datasets (70B or 8B)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate for training")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for optimizer")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                      help="Dropout rate for MLP head")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--checkpoint_frequency", type=int, default=None,
                      help="Save a checkpoint every N steps (e.g., 10000)")
    
    # Data parameters
    parser.add_argument("--dataset_path", type=str, default=None,
                      help="HuggingFace dataset path")
    parser.add_argument("--all_datasets", action="store_true",
                      help="Use all three hardcoded datasets (MATH500, GPQA, and MMLU-Pro)")
    parser.add_argument("--score_columns", type=str, nargs="+", default=None,
                      help="Columns to use for scores")
    parser.add_argument("--use_naive_ensemble", action="store_true",
                      help="Whether to use naive ensemble by averaging all verdict and score columns")
    parser.add_argument("--max_rows", type=int, default=None,
                      help="Maximum number of rows to process")
    parser.add_argument("--start_row", type=int, default=None,
                      help="Starting row index for training")
    parser.add_argument("--end_row", type=int, default=None,
                      help="Ending row index for training (exclusive)")
    parser.add_argument("--shuffle_samples", action="store_true",
                      help="Whether to shuffle samples within each row")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                      help="Output directory for checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Whether to use wandb logging")
    
    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                      help="Number of epochs to wait before early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.00000001,
                      help="Minimum change in validation loss to be considered as improvement")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    args = parser.parse_args()
    
    # Convert mlp_hidden_dims from string to list of integers if provided
    if args.mlp_hidden_dims is not None:
        args.mlp_hidden_dims = [int(dim) for dim in args.mlp_hidden_dims.split(',')]
    
    # Create config
    config = TrainingConfig(**vars(args))
    
    # Train model
    trainer = CrossEncoderTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
