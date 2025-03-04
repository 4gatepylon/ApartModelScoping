import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, TrainerCallback
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import sys
import os
from huggingface_hub import InferenceClient, login
import copy
import random

from dotenv import load_dotenv
load_dotenv(dotenv_path="/om/user/dbaek/ApartModelScoping/.env")

from pathlib import Path
import argparse

from dotenv import load_dotenv
load_dotenv()

import wandb
import numpy as np


seed = 37
# Torch seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

current_path = Path(__file__)
sys.path.append(str(current_path.parent.parent))

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name1 = "meta-llama/Llama-3.2-1B-Instruct"
cache_dir = "/om2/user/dbaek/MODELS"
tokenizer = AutoTokenizer.from_pretrained(model_name1, cache_dir=cache_dir)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model1 = AutoModelForCausalLM.from_pretrained(model_name1, cache_dir=cache_dir)

# ====================================================
# Freeze all parameters first.
# ====================================================
for param in model1.parameters():
    param.requires_grad = False

# ====================================================
# Determine the middle two layers to fine-tune.
# ====================================================
n_layers = len(model1.model.layers)
if n_layers < 2:
    raise ValueError("Not enough layers to fine-tune two middle layers.")

indices_to_unfreeze = [n_layers // 3]

# Unfreeze only the two middle layers.
for i in indices_to_unfreeze:
    for name, param in model1.model.layers[i].named_parameters():
        param.requires_grad = True

# Optionally, print out the trainable parameters from the chosen layers.
print("Trainable parameters in the middle layers:")
for i in indices_to_unfreeze:
    print(f"Layer {i}:")
    for name, param in model1.model.layers[i].named_parameters():
        if param.requires_grad:
            print("  ", name)

# --------------------------
# Randomly Zero Out a Set of Weights (only affects trainable parameters)
# --------------------------
def zero_out_random_weights(model, zero_percent=0.1):
    """
    Randomly sets a percentage (zero_percent) of each trainable parameter to zero.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask = torch.rand_like(param) < zero_percent
            param.data[mask] = 0
            print(f"Zeroed out {mask.sum().item()} elements in parameter: {name}")

# Zero out 10% of the trainable weights (adjust as needed)
zero_out_random_weights(model1, zero_percent=0.1)

model1 = model1.to(device)

dataset_name = "camel-ai/biology"
ctx_len = 512

train_data = load_dataset(dataset_name, split="train", cache_dir="/home/dbaek/.cache/")
train_data = train_data.shuffle(seed=37)

def tokenize_function(examples):
    # Process each example in the batch.
    texts = [
        (m1 + "\n" + m2)  
        for m1, m2 in zip(examples['message_1'], examples['message_2'])
    ]
    sys.stdout.flush()
    tok = tokenizer(texts, padding=False, truncation=True, max_length=ctx_len)
    print(np.array(tok['input_ids']).shape)
    sys.stdout.flush()
    return tok

def filter_examples(example):
    tokens = tokenizer.tokenize(
        example['message_1'] + '\n' + example['message_2'],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=ctx_len
    )
    return len(tokens) >= ctx_len


train_tokenized_datasets = train_data.filter(filter_examples).map(tokenize_function, batched=True)

def trainer_compute_loss(model, inputs, return_outputs=False):
    input_ids = inputs["input_ids"].to(device)
    if input_ids.dim() != 2:
        input_ids = input_ids.unsqueeze(0)
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf detected in parameters of layer {name}")
    outputs = model(input_ids=input_ids)
    logits = outputs['logits']

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return (loss, outputs) if return_outputs else loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(device)
        if input_ids.dim() != 2:
            input_ids = input_ids.unsqueeze(0)
                
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(loss)
        sys.stdout.flush()

        return (loss, outputs) if return_outputs else loss

class TrainLossConvergenceCallback(TrainerCallback):
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.num_bad_steps = 0

    def on_step_end(self, args, state, control, **kwargs):
        if len(state.log_history) < 10:
            return
        tmp_list = []
        for i in range(1,11):
            tmp_list.append(state.log_history[-i]["loss"])
        current_loss = sum(tmp_list) / len(tmp_list)
        
        if self.best_loss - current_loss >= self.threshold:
            self.best_loss = current_loss
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        print(current_loss, self.num_bad_steps)
        sys.stdout.flush()

        if self.num_bad_steps >= self.patience:
            print(f"Stopping training early due to no improvement in training loss after {self.patience} steps.")
            control.should_training_stop = True

wandb.init(project="Scoping")
training_args = TrainingArguments(
    output_dir=f"./results/",
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=1,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=10000,
    run_name=f"scoping-{dataset_name}-{model_name1}-seed{seed}",
    report_to=["wandb"]
)

trainer = CustomTrainer(
    model=model1,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    callbacks=[TrainLossConvergenceCallback(patience=50, threshold=0.001)],
)

# Train the model
train_output = trainer.train()

final_loss = train_output.training_loss
print(train_output)
print(f"Final training loss: {final_loss}")
sys.stdout.flush()

tokenizer.save_pretrained(f"./results/")