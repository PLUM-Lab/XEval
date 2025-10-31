# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
import numpy as np
import math
import os
import sys
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping, Sequence
from pathlib import Path
import pandas as pd
import logging

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, set_seed
from transformers.trainer_pt_utils import LabelSmoother
from transformers.testing_utils import CaptureLogger
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
import pdb
from functools import partial

from longchat.conversation import get_default_conv_template, SeparatorStyle
from longchat.train.safe_save_trainer import SafeSaveTrainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
HUGGING_FACE_TOKEN = ""


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})
    train_dir: str = field(
        default=None, metadata={"help": "Path to the training data dir."}
    )
    valid_dir: str = field(
        default=None, metadata={"help": "Path to the valid data dir."}
    )
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    num_preprocess_worker: int = field(
        default=96, metadata={"help": "Number of cpus to tokenize and process datasets."}
    )
    hf_arrow_data_path: str = field(
        default=None, metadata={"help": "Path to the saved hf arrow data dir."}
    )
    lazy_preprocess: bool = False
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    validation_split_percentage: Optional[float] = field(
        default=0.0001,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cached dataset."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def encode_with_pretrain_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'text' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    example_text = example['text']
    # + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, 
                                  return_tensors='pt', 
                                  max_length=max_seq_length, 
                                  truncation=True, 
                                  padding="max_length"
                        )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    attention_mask = tokenized_example.attention_mask
    assert input_ids.shape[1] == max_seq_length
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

logger = logging.getLogger(__name__)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)
    
    if True:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    set_seed(training_args.seed)
    
    # initialize our customized tokenizer
    special_tokens = [ f"<IMG-CODE-{i}>" for i in range(16384) ]
    special_tokens = ["<_IMG-BEG_>"] + special_tokens + ['<_IMG-END_>']
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        additional_special_tokens=special_tokens,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    # make sure the token size match
    assert len(tokenizer) == 16386 + 32000

    # set block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = data_args.block_size
    logger.info(f'Using block size {block_size} for chunking the training data')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    model.config.use_cache = False  # required by flash attention
    # resize the embeddings
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        # with CaptureLogger(tok_logger) as cl:
        # we hope to add special eos token
        examples["text"] = [ ex_text+tokenizer.eos_token for ex_text in examples["text"]]
        output = tokenizer(examples["text"])
        # for ids_, mask in zip(output.input_ids, output.attention_mask):
        #     print(len(ids_),len(mask))
        # print(len(examples["text"]))
        # print(len(output.input_ids[0]), output.input_ids[1])
        # clm input could be much much longer than block_size
        # if "Token indices sequence length is longer than the" in cl.out:
        #     tok_logger.warning(
        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
        #         " before being passed to the model."
        #     )
        return output
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        logger.info(f"total lenght of 1000 examples is {total_length}")
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="# --------- loading dataset and tokenization ------------- #"):
        #  first let's read dataset from each file
        if not data_args.use_cache:
            path = Path(data_args.dataset_dir)
            files = [file.name for file in path.glob("*.jsonl")]
            for idx, file in enumerate(files):
                data_file = os.path.join(path, file)
                filename = ''.join(file.split(".")[:-1])
                cache_path = os.path.join(data_args.data_cache_dir, filename)
                os.makedirs(cache_path, exist_ok=True)
                try:
                    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                    logger.info(f'training datasets-{filename} has been loaded from disk')
                except Exception:
                    cache_dir = os.path.join(data_args.data_cache_dir, filename+"_text")
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info(f"loading data from {data_file}")
                    raw_dataset = load_dataset("json", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                    logger.info(f"{file} has been loaded from json file")
                    tokenized_dataset = raw_dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.num_preprocess_worker,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        remove_columns=["text",'id_'],
                        cache_file_names = {k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                        desc="Running tokenizer on dataset",
                    )
                    logger.info(f"{file} has been tokenized")
                    grouped_datasets = tokenized_dataset.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.num_preprocess_worker,
                        load_from_cache_file=True,
                        keep_in_memory=False,
                        cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                        desc=f"Grouping texts in chunks of {block_size}",
                    )
                    logger.info(f"{file} has been chunked")
                    processed_dataset = grouped_datasets
                    processed_dataset.save_to_disk(cache_path)
                    # logger.info(f"{file} generates {len(processed_dataset)} training instances.")
                if idx == 0:
                    lm_datasets = processed_dataset['train']
                else:
                    assert lm_datasets.features.type == processed_dataset["train"].features.type
                    lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
            lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)
            lm_datasets.save_to_disk(data_args.data_cache_dir)
        else:
            logger.info(f"# -------- Use cached dataset and load data from {data_args.data_cache_dir} ----------- #")
            lm_datasets = datasets.load_from_disk(data_args.data_cache_dir, keep_in_memory=False)
            logger.info(f'training datasets has been loaded from disk')
            
        logger.info(f"Total data set size is {len(lm_datasets)}")
        # sample the amount of training data for validation


    
    # -------------------- start training -----------------------
    train_dataset = lm_datasets['train']
    logger.info(f"Num train_samples  {len(train_dataset)}")
    logger.info("training example:")
    logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("testing example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
    else:
        eval_dataset = None

    logger.info("#------------- Finish data preprocessing and start training ---------#")
    trainer = SafeSaveTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    logger.info("#------------- Finish Training ---------#")
    trainer.save_state()
    # trainer.save_model(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) # cause memory problem


if __name__ == "__main__":
    train()
