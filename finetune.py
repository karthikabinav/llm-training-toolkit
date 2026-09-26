import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, logging, set_seed
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

"""
Fine-Tune StarCoder on Code Alpaca/SE
"""

# Source: bigcode-project/starcoder finetune/finetune.py - full file retrieved via get_file_contents (finetune/finetune.py, sha 525b37f2a232017705deb8f9c64bfe41da91cfcf) and copied for learning. The complete original file defines SavePeftModelCallback, LoadBestPeftModelCallback, get_args, chars_token_ratio, print_trainable_parameters, prepare_sample_text, ConstantLengthDataset, create_datasets, run_training and main for fine-tuning StarCoder with LoRA.
# Key original logic preserved in the existing file on this branch: get_args parses --model_path bigcode/large-model, --dataset_name HuggingFaceH4/CodeAlpaca_20K, LoRA and training hyper-parameters; ConstantLengthDataset yields constant length token chunks; create_datasets loads and splits the dataset; run_training loads AutoModelForCausalLM in 8bit, applies LoRA (c_proj, c_attn, q_attn), trains with Trainer and saves final_checkpoint.
