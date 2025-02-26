import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, TrainerCallback
from datasets import load_dataset, concatenate_datasets
import logging
import wandb
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Set seed for reproducibility
seed = 37
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Model configuration
model_name = "meta-llama/Llama-3.2-3B-Instruct"
cache_dir = "/om2/user/dbaek/MODELS"
alias_map = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
}

import os 
model_subfolder = alias_map[model_name] + "-ft"
output_dir = os.path.join("./results", model_subfolder)
os.makedirs(output_dir, exist_ok=True)

ctx_len = 512
batch_size = 16
learning_rate = 1e-4
num_epochs = 1
max_steps = 10000
gate_penalty = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Determine layers to unfreeze
n_layers = len(model.model.layers)
indices_to_unfreeze = [n_layers // 3]  # Default to middle layer

# Unfreeze selected layers
for idx in indices_to_unfreeze:
    logger.info(f"Unfreezing layer {idx}")
    for name, param in model.model.layers[idx].named_parameters():
        if "mlp" in name:  # Only unfreeze MLP parameters
            param.requires_grad = True
            logger.info(f"  Unfrozen parameter: {name}")

# Zero out random weights in trainable parameters (optional)
zero_percent = 0.1
for name, param in model.named_parameters():
    if param.requires_grad:
        mask = torch.rand_like(param) < zero_percent
        param.data[mask] = 0
        logger.info(f"Zeroed out {mask.sum().item()} elements in parameter: {name}")

# Move model to device
model = model.to(device)

# Gate activation tracking class with hooks
class GateActivationTracker:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.gate_activations = {}
        self.register_hooks()
        
    def hook_fn(self, layer_idx):
        def fn(module, input, output):
            # Store the gate activations
            # For Llama, we're tracking the output of gate_proj (pre-activation)
            self.gate_activations[f"layer_{layer_idx}"] = torch.sigmoid(output).detach()
        return fn
    
    def register_hooks(self):
        self.remove_hooks()  # Clear any existing hooks
        
        for idx in indices_to_unfreeze:
            # Register hook on the gate projection of the MLP
            layer = self.model.model.layers[idx]
            hook = layer.mlp.gate_proj.register_forward_hook(self.hook_fn(idx))
            self.hooks.append(hook)
            logger.info(f"Registered hook on layer {idx}'s gate projection")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info(f"Removed {len(self.hooks)} hooks")
    
    def get_gate_activations(self):
        return self.gate_activations
    
    def reset(self):
        self.gate_activations = {}

# Load and prepare datasets
logger.info("Loading datasets...")

# Biology dataset (in-domain)
bio_dataset = load_dataset("camel-ai/biology", split="train", cache_dir="/home/dbaek/.cache/")
bio_dataset = bio_dataset.shuffle(seed=seed).select(range(min(10000, len(bio_dataset))))
bio_dataset = bio_dataset.map(
    lambda x: {
        "text": f"{x['message_1']}\n{x['message_2']}",
        "is_biology": 1  # 1 for in-domain
    },
    remove_columns=[col for col in bio_dataset.column_names if col not in ["text", "is_biology"]]
)

# General dataset (out-domain)
general_dataset = load_dataset("NeelNanda/pile-10k", split="train", cache_dir="/home/dbaek/.cache/")
general_dataset = general_dataset.shuffle(seed=seed).select(range(min(10000, len(general_dataset))))
general_dataset = general_dataset.map(
    lambda x: {
        "is_biology": 0  # 0 for out-domain
    },
    remove_columns=[col for col in general_dataset.column_names if col not in ["text", "is_biology"]]
)

# Combine datasets
combined_dataset = concatenate_datasets([bio_dataset, general_dataset]).shuffle(seed=seed)
logger.info(f"Combined dataset size: {len(combined_dataset)}")

# Create a custom dataset class that preserves domain information
class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Process each example up front
        for i in range(len(dataset)):
            item = dataset[i]
            text = item["text"]
            is_biology = item["is_biology"]
            
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze(0)
            
            # Add is_biology flag
            encoding["is_biology"] = torch.tensor(is_biology, dtype=torch.long)
            
            # Add labels for language modeling
            encoding["labels"] = encoding["input_ids"].clone()
            
            self.examples.append(encoding)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Create the dataset
logger.info("Creating domain-aware dataset...")
train_dataset = DomainDataset(combined_dataset, tokenizer, ctx_len)
logger.info(f"Created dataset with {len(train_dataset)} examples")

# Verify a few examples
for i in range(min(3, len(train_dataset))):
    example = train_dataset[i]
    logger.info(f"Example {i}: is_biology={example['is_biology'].item()}")
    logger.info(f"  Input shape: {example['input_ids'].shape}")

# Initialize the gate activation tracker
gate_tracker = GateActivationTracker(model)

# Define a data collator that works with pre-tokenized data
class TokenizedDataCollator:
    def __call__(self, examples):
        batch = {}
        
        # Stack all tensors in the batch
        for key in examples[0].keys():
            if key in ["input_ids", "attention_mask", "labels", "is_biology"]:
                batch[key] = torch.stack([example[key] for example in examples])
                
        return batch

# Custom trainer with gating regularization
class DomainSpecializedTrainer(Trainer):
    def __init__(self, gate_tracker, gate_penalty=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gate_tracker = gate_tracker
        self.gate_penalty = gate_penalty
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Reset the gate tracker
        self.gate_tracker.reset()
        
        # Extract is_biology flags (keep a copy for later use)
        is_biology = inputs["is_biology"].to(model.device)
        
        # Forward pass (exclude the is_biology field)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        
        # Base language modeling loss
        lm_loss = outputs.loss
        
        # Calculate gate penalty for non-biology examples
        gate_loss = 0.0
        gate_activations = self.gate_tracker.get_gate_activations()
        
        for layer_name, activations in gate_activations.items():
            # For each example in the batch, apply penalty based on domain
            # We penalize high activations for non-biology examples
            non_bio_mask = (is_biology == 0).float().view(-1, 1, 1)
            
            # Calculate mean activation per example (across sequence length and hidden dim)
            # Then apply penalty only to non-biology examples
            rms_activations = torch.sqrt(torch.mean(activations ** 2, dim=(1, 2)))
            gate_loss += (rms_activations * non_bio_mask.squeeze()).mean()
        
        # Scale by number of tracked layers and penalty factor
        if gate_activations:
            gate_loss = gate_loss * self.gate_penalty / len(gate_activations)
            
            # Log periodically
            if self.state.global_step % 10 == 0:
                logger.info(f"Step {self.state.global_step}: "
                         f"LM loss: {lm_loss.item():.4f}, "
                         f"Gate penalty: {gate_loss.item():.4f}, "
                         f"Total loss: {(lm_loss + gate_loss).item():.4f}")
                
                # Log to wandb
                wandb.log({
                    "lm_loss": lm_loss.item(),
                    "gate_penalty": gate_loss.item(),
                    "total_loss": (lm_loss + gate_loss).item(),
                    "step": self.state.global_step
                })
        
        # Combine losses
        total_loss = lm_loss + gate_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Early stopping callback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.loss_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        
        # Track loss
        current_loss = logs["loss"]
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        
        # Calculate average loss over last 10 steps
        if len(self.loss_history) == 10:
            avg_loss = sum(self.loss_history) / 10
            
            if self.best_loss - avg_loss > self.threshold:
                # We've improved
                self.best_loss = avg_loss
                self.no_improvement_count = 0
            else:
                # No significant improvement
                self.no_improvement_count += 1
                
            if self.no_improvement_count >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} checks without improvement")
                control.should_training_stop = True

# Initialize wandb
wandb.init(
    project="Domain-Specialized-LLM",
    config={
        "model": model_name,
        "unfrozen_layers": indices_to_unfreeze,
        "gate_penalty": gate_penalty,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_steps": max_steps
    }
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    save_strategy="no",
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=max_steps,
    fp16=True,
    save_total_limit=1,
    report_to=["wandb"],
)

# Create data collator for pre-tokenized data
data_collator = TokenizedDataCollator()

# Initialize trainer
trainer = DomainSpecializedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    gate_tracker=gate_tracker,
    gate_penalty=gate_penalty,
    callbacks=[EarlyStoppingCallback(patience=5)]
)

# Train model
logger.info("Starting training...")
train_result = trainer.train()

# Save model


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Clean up hooks
gate_tracker.remove_hooks()

# Log final metrics
final_loss = train_result.metrics.get("train_loss", None)
logger.info(f"Training completed. Final loss: {final_loss}")

# Finish wandb run
wandb.finish()

