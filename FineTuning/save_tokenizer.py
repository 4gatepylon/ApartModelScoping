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

tokenizer.save_pretrained("./results/checkpoint-458/")