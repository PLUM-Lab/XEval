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
from typing import Dict, Optional, Sequence
import pdb

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from longchat.conversation import get_default_conv_template, SeparatorStyle
from longchat.train.safe_save_trainer import SafeSaveTrainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
HUGGING_FACE_TOKEN = "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"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_grad_norm=0.5


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


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """ The template for training data is:
        conv_vicuna_v1_1 = Conversation(
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Below is a record of our previous conversation on 10 different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say 'I would like to discuss the topic of <TOPIC>'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first topic. Now the record start. USER: I would like to discuss the topic of the history and culture of the Middle Ages. 
    ASSISTANT: Sure, I'd be happy to discuss that topic with you. What would you like to know about it? 
    USER: I'm curious about what life was like during the Middle Ages, and how it differed from modern times.
    """
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # map human to USER and gpt to ASSISTANT

    # Apply prompt templates
    conversations = []
    """
    sources = {
                "id": "identity_0",
                "conversations": [
                    {
                    "from": "human",
                    "value": "Who are you?"
                    },
                    {
                    "from": "gpt",
                    "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
                    },
                    {
                    "from": "human",
                    "value": "What can you do?"
                    },
                    {
                    "from": "gpt",
                    "value": "I can chat with you."
                    }
                ]
            }
    """
    for i, source in enumerate(sources):
        #if source[0]["from"] not in roles.keys() or roles[source[0]["from"]] != conv.roles[0]:
        if  roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        skipped = 0
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[(j - skipped) % 2]:
                print("skipping misaligned rounds")
                skipped += 1
                continue # skipp if two rounds are from the user or two round are from assistant
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # rank0_print(conversations)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    
    # unlimited_input_ids = tokenizer(
    #     conversations,
    #     return_tensors="pt"
    # ).input_ids
    
    # print(unlimited_input_ids.shape, input_ids.shape)

    assert conv.sep_style == SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID # mask the turns of human in the target
            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(
                    f"WARNING: tokenization mismatch " f"{cur_len} vs. {total_len}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        """ input data format: list[dict]
        [
            {
            "id": "identity_0",
            "conversations": [
                {
                "from": "human",
                "value": "Who are you?"
                },
                {
                "from": "gpt",
                "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
                },
                {
                "from": "human",
                "value": "What can you do?"
                },
                {
                "from": "gpt",
                "value": "I can chat with you."
                }
            ]
            },
            {
            "id": "identity_1",
            "conversations": [
                {
                "from": "human",
                "value": "Who are you?"
                },
                {
                "from": "gpt",
                "value": "My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS)."
                }
            ]
            }
        ]
        """
        print(len(list_data_dict))
        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        print(len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess([e["conversations"] for e in sources], self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                attention_mask=data_dict["attention_mask"][0],
            )
        return data_dict


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    #import os
    #os.environ["WANDB_DISABLED"] = "true"
    trainer = SafeSaveTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()