# -*- coding: utf-8 -*-
import os
import sys
import logging
from datasets import load_dataset, concatenate_datasets
import transformers
import pandas as pd
from itertools import chain
import multiprocessing as mp
import random 
from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)

sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[pad]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def prebuild_tokenizer(tokenizer, model=None):
    origin_tokenizer_len = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    new_tokenizer_len = len(tokenizer)
    if origin_tokenizer_len != new_tokenizer_len and model:
        print(f"resize embeddings from {origin_tokenizer_len} to {new_tokenizer_len}")
        model.resize_token_embeddings(new_tokenizer_len)




class ConversationPrompter:
    SYSTEM_PROMPT = 'System:\n'
    HUMAN_PROMPT = '\n\nHuman:\n'
    AI_PROMPT = '\n\nAssistant:\n'
    SYSTEM_PROMPT_ZH = '系统:\n'
    HUMAN_PROMPT_ZH = '\n\n人类:\n'
    AI_PROMPT_ZH = '\n\nAI助手:\n'
    HUMAN = ('human', 'user')
    # {'human': 4438666, 'gpt': 413497, 'bing': 128, 'chatgpt': 427, 'bard': 8, 'assistant': 4024539}
    AI = ('ai', 'assistant', 'bing', 'gpt', 'bard', 'chatgpt', 'claude')
    SYS = 'system'
    def __init__(self, tokenizer, train_on_inputs=False):
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
    
    def from_human(self, speaker: str):
        return speaker.lower().startswith(self.HUMAN)

    def from_ai(self, speaker: str):
        return speaker.lower().startswith(self.AI)

    def from_system(self, speaker: str):
        return speaker.lower().startswith(self.SYS)
        

    def tokenize_system(self, content, add_special_tokens=True):
        inputs_ids = self.tokenizer.encode(self.SYSTEM_PROMPT + content, add_special_tokens=add_special_tokens)
        if not self.train_on_inputs:
            labels = [-100] * len(inputs_ids)
        else:
            labels = inputs_ids.copy()
        return {"input_ids": inputs_ids, "labels": labels}
    
    def tokenize_human(self, content, add_special_tokens):
        human_prompt = self.HUMAN_PROMPT if not add_special_tokens else self.HUMAN_PROMPT.lstrip()       
        prompt = human_prompt + content 
        inputs_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if not self.train_on_inputs:
            labels = [-100] * len(inputs_ids)
        else:
            labels = inputs_ids.copy()
        return {"input_ids": inputs_ids, "labels": labels}
    
    def tokenize_ai(self, content, add_special_tokens=False):
        ai_prompt = self.AI_PROMPT if not add_special_tokens else self.AI_PROMPT.lstrip()
        ai_prompt_len = len(self.tokenizer.encode(ai_prompt, add_special_tokens=False)) 
        prompt = ai_prompt + content
        inputs_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        labels = inputs_ids.copy()
        if not self.train_on_inputs:
            # add eos token
            inputs_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
            labels[:ai_prompt_len] = [-100] * ai_prompt_len
        return {"input_ids": inputs_ids, "labels": labels}

    def tokenize_one_turn(self, speaker, content, add_special_tokens=False):
        if self.from_human(speaker):
            return self.tokenize_human(content, add_special_tokens=add_special_tokens)
        elif self.from_ai(speaker):
            return self.tokenize_ai(content, add_special_tokens=add_special_tokens)
        elif self.from_system(speaker):
            return self.tokenize_system(content, add_special_tokens=add_special_tokens)
        else:
            raise AssertionError(f'Not supported speaker:{speaker}')
   



def load_tokenized_conversation_dataset(    
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_paths: list[str],
    val_set_size: int,
    cutoff_len: int = 512,
    train_on_inputs: bool = False,
    select_samples: None | list = None,
    complete_alpha: float = 0.6,
):
    prompter = ConversationPrompter(tokenizer, train_on_inputs=train_on_inputs)
    def generate_and_tokenize_prompt_mask_input(example):
        conversations = example["conversations"]
        inputs_ids = [] 
        labels = []
        speakers = []
        used_len = 0
        attention_mask = []
        # if system_prompt is not empty, we add it to the beginning
        if example.get("system_prompt", ''):
            inputs = prompter.tokenize_system(example["system_prompt"], True)
            inputs_ids.append(inputs["input_ids"])
            labels.append(inputs["labels"]) 
            speakers.append("system_prompt")
            used_len += len(inputs["input_ids"])

        for turn in conversations:
            speaker = turn["from"]
            content = turn["value"]
            inputs = prompter.tokenize_one_turn(speaker, content, add_special_tokens=len(inputs_ids)==0)
            current_idx = inputs["input_ids"]
            current_label = inputs["labels"]
            # If left space is not enough for the complete_alpha percent of current input, we drop it.
            if cutoff_len - used_len < complete_alpha * len(current_idx):
                break
            inputs_ids.append(current_idx)
            labels.append(current_label)
            speakers.append(speaker)
            used_len += len(current_idx)
        # if the last speaker is human, we drop it
        if len(speakers) > 0  and prompter.from_human(speakers[-1]):
            inputs_ids = inputs_ids[:-1]
            labels = labels[:-1]
            speakers = speakers[:-1]
        # change to 1-D list
        inputs_ids = list(chain(*inputs_ids))
        labels = list(chain(*labels))
        # it would happen when the last response is long enough
        if len(inputs_ids) >= cutoff_len:
            inputs_ids = inputs_ids[:cutoff_len]
            labels = labels[:cutoff_len]
        if len(inputs_ids) > 0 and inputs_ids[-1] != tokenizer.eos_token_id:
            # definitely not been cutted
            if len(inputs_ids) < cutoff_len:
                inputs_ids.append(tokenizer.eos_token_id)
                # learn eos token only from response
                labels.append(-100 if prompter.from_human(speakers[-1]) or prompter.from_system(speakers[-1]) else tokenizer.eos_token_id)
            # cutted to cutoff_len or happen to equal cutoff_len
            else:
                inputs_ids[-1] = tokenizer.eos_token_id
                # not the last token, we set to -100
                labels[-1] = -100
        # all label ids are -100, we set to empty and filter later
        if labels.count(-100) == len(labels):
            inputs_ids = []
        attention_mask = [1] * len(inputs_ids)
        return {"input_ids": inputs_ids, "labels": labels, "attention_mask": attention_mask}


    data = load_conversation_dataset_from_paths(dataset_paths)
    print(len(data))
    if len(data) == 0:
        raise AssertionError(f'Empty dataset with sample number 0. Please check the dataset paths: {dataset_paths}')

    if val_set_size > 0:
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (
            train_val["train"]
            .shuffle()
            .map(
                generate_and_tokenize_prompt_mask_input,
                num_proc=mp.cpu_count() - 2,
                remove_columns=data.column_names,
            )
            .filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt_mask_input, remove_columns=data.column_names)
            .filter(lambda x: len(x["input_ids"]) > 0)
        )
    else:
        # for testing
        if select_samples is not None and len(select_samples) > 0:
            train_data = data.select(select_samples).map(
                generate_and_tokenize_prompt_mask_input, remove_columns=data.column_names
            ).filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
        else:
            train_data = data.shuffle().map(
                generate_and_tokenize_prompt_mask_input,
                num_proc=mp.cpu_count() - 2,
                remove_columns=data.column_names,
            ).filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
            
        val_data = None

    return train_data, val_data
        

        


def load_conversation_dataset_from_paths(
    dataset_paths: list[str],
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    raw_datasets_list = []
    for i_data, dataset_path in enumerate(dataset_paths):
        # get file extentions
        extention = dataset_path.split(".")[-1]
        extention = "json" if extention == "jsonl" else extention
        # verify data file
        assert extention in [
            "csv",
            "json",
            "jsonl",
            "parquet",
        ], f"File type error: {dataset_path}"
        assert os.path.isfile(dataset_path), f"File: {dataset_path} not exists."
        # read data
        raw_dataset = load_dataset(extention, data_files=dataset_path, split="train")
        logger.info(
            f"\n\nLoad dataset: {i_data+1}/{len(dataset_paths)}\nFrom path: {dataset_path}\nColumn names:{raw_dataset.column_names}\n"
        )
        # standard format
        if len({"conversations"} - set(raw_dataset.column_names)) == 0:
            pass
        else:
            raise KeyError(
                f"Data format error: {dataset_path}, columns: {raw_dataset.column_names}"
            )
        raw_datasets_list.append(raw_dataset)

    if len(raw_datasets_list) == 1:
        raw_datasets = raw_datasets_list[0]
    else:
        raw_datasets = concatenate_datasets(raw_datasets_list)
    assert len({"conversations"} - set(raw_datasets.column_names)) == 0, raw_datasets.column_names
    raw_datasets = raw_datasets.filter(
        lambda example: len(example["conversations"])> 1 and example["conversations"][0]["from"].lower() in ["human", "user"] ,
        num_proc=mp.cpu_count() -1,
        desc="Remove empty conversations",
    )
    return raw_datasets




