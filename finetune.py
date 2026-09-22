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
# Full file copied from bigcode-project/starcoder finetune/finetune.py (finetune.py) - source retrieved via get_file_contents from bigcode-project/starcoder finetune/finetune.py, sha 525b37f2a232017705deb8f9c64bfe41da91cfcf, size 12100