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


def prebuild_tokenizer(tokenizer, model=None, padding_side="right"):
    origin_tokenizer_len = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2
    tokenizer.padding_side = padding_side
    new_tokenizer_len = len(tokenizer)
    if origin_tokenizer_len != new_tokenizer_len and model:
        print(f"resize embeddings from {origin_tokenizer_len} to {new_tokenizer_len}")
        model.resize_token_embeddings(new_tokenizer_len)



class ConversationPrompter:
    HUMAN = ('human', 'user')
    # {'human': 4438666, 'gpt': 413497, 'bing': 128, 'chatgpt': 427, 'bard': 8, 'assistant': 4024539}
    AI = ('ai', 'assistant', 'bing', 'gpt', 'bard', 'chatgpt', 'claude')
    SYS = 'system'
    def __init__(self, tokenizer, train_on_inputs=False, cutoff_len=512, complete_alpha=0.6, system_prompt_tmp = 'System:\n', human_prompt_tmp = '\n\nHuman:\n', ai_prompt_tmp = '\n\nAssistant:\n'):
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.cutoff_len = cutoff_len
        self.complete_alpha = complete_alpha
        self.system_prompt = system_prompt_tmp
        self.human_prompt = human_prompt_tmp
        self.ai_prompt = ai_prompt_tmp
    
    def from_human(self, speaker: str):
        return speaker.lower().startswith(self.HUMAN)

    def from_ai(self, speaker: str):
        return speaker.lower().startswith(self.AI)

    def from_system(self, speaker: str):
        return speaker.lower().startswith(self.SYS)
        

    def tokenize_system(self, content, add_special_tokens=True):
        inputs_ids = self.tokenizer.encode(self.system_prompt + content, add_special_tokens=add_special_tokens)
        if not self.train_on_inputs:
            labels = [-100] * len(inputs_ids)
        else:
            labels = inputs_ids.copy()
        return {"input_ids": inputs_ids, "labels": labels}
    
    def tokenize_human(self, content, episode_num, add_special_tokens):
        human_prompt = self.human_prompt if not add_special_tokens else self.human_prompt.lstrip()       
        prompt = human_prompt + content 
        inputs_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if not self.train_on_inputs:
            labels = [-100] * len(inputs_ids)
        else:
            labels = inputs_ids.copy()
        return {"input_ids": inputs_ids, "labels": labels}
    
    def tokenize_ai(self, content, episode_num, add_special_tokens=False):
        ai_prompt = self.ai_prompt if not add_special_tokens else self.ai_prompt.lstrip()
        ai_prompt_len = len(self.tokenizer.encode(ai_prompt, add_special_tokens=False)) 
        prompt = ai_prompt + content
        inputs_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        labels = inputs_ids.copy()
        # add eos token
        inputs_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        if not self.train_on_inputs:
            labels[:ai_prompt_len] = [-100] * ai_prompt_len
        return {"input_ids": inputs_ids, "labels": labels}

    def tokenize_one_turn(self, speaker, content, episode_num, add_special_tokens=False):
        if self.from_human(speaker):
            return self.tokenize_human(content, episode_num=episode_num, add_special_tokens=add_special_tokens)
        elif self.from_ai(speaker):
            return self.tokenize_ai(content, episode_num=episode_num, add_special_tokens=add_special_tokens)
        elif self.from_system(speaker):
            return self.tokenize_system(content, add_special_tokens=add_special_tokens)
        else:
            raise AssertionError(f'Not supported speaker:{speaker}')

    
    def generate_and_tokenize_prompt_mask_input(self, example):
        conversations = example["conversations"]
        inputs_ids = [] 
        labels = []
        speakers = []
        used_len = 0
        attention_mask = []
        # if system_prompt is not empty, we add it to the beginning
        if example.get("system_prompt", ''):
            inputs = self.tokenize_system(example["system_prompt"], True)
            inputs_ids.append(inputs["input_ids"])
            labels.append(inputs["labels"]) 
            speakers.append("system_prompt")
            used_len += len(inputs["input_ids"])

        for episode_num, turn in enumerate(conversations):
            speaker = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            # only add sepicial token at the beginning
            inputs = self.tokenize_one_turn(speaker, content, episode_num=episode_num, add_special_tokens=len(inputs_ids)==0)
            current_idx = inputs["input_ids"]
            current_label = inputs["labels"]
            # If left space is not enough for the complete_alpha percent of current input, we drop it.
            if self.cutoff_len - used_len < self.complete_alpha * len(current_idx):
                break
            inputs_ids.append(current_idx)
            labels.append(current_label)
            speakers.append(speaker)
            used_len += len(current_idx)
        # if the last speaker is human, we drop it
        if len(speakers) > 0  and self.from_human(speakers[-1]):
            inputs_ids = inputs_ids[:-1]
            labels = labels[:-1]
            speakers = speakers[:-1]
        # change to 1-D list
        inputs_ids = list(chain(*inputs_ids))
        labels = list(chain(*labels))
        # it would happen when the last response is long enough
        if len(inputs_ids) >= self.cutoff_len:
            inputs_ids = inputs_ids[:self.cutoff_len]
            labels = labels[:self.cutoff_len]
        if len(inputs_ids) > 0 and inputs_ids[-1] != self.tokenizer.eos_token_id:
            # definitely not been cutted
            if len(inputs_ids) < self.cutoff_len:
                inputs_ids.append(self.tokenizer.eos_token_id)
                # learn eos token only from response
                labels.append(-100 if self.from_human(speakers[-1]) or self.from_system(speakers[-1]) else self.tokenizer.eos_token_id)
            # cutted to cutoff_len or happen to equal cutoff_len
            else:
                pass
            #     inputs_ids[-1] = self.tokenizer.eos_token_id
            #     # not the last token, we set to -100
            #     labels[-1] = -100
        # all label ids are -100, we set to empty and filter later
        if labels.count(-100) == len(labels):
            inputs_ids = []
        attention_mask = [1] * len(inputs_ids)
        return {"input_ids": inputs_ids, "labels": labels, "attention_mask": attention_mask}
   
class MistralPrompter(ConversationPrompter):
    """["<s> [INST] What is your favourite condiment? [/INST]Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s>  [INST] Do you have mayonnaise recipes? [/INST]"]"""
    def __init__(self, tokenizer, train_on_inputs=False, cutoff_len=512, complete_alpha=0.6, system_prompt_tmp = '[INST] ', human_prompt_tmp = ' [/INST]', ai_prompt_tmp = ''):
        super().__init__(tokenizer, train_on_inputs, cutoff_len, complete_alpha, system_prompt_tmp=system_prompt_tmp, human_prompt_tmp=human_prompt_tmp, ai_prompt_tmp=ai_prompt_tmp)

            
    def tokenize_human(self, content, episode_num, add_special_tokens):
        # there is a system_prompt exist
        if  episode_num == 0 and not add_special_tokens:
            prompt = content + self.human_prompt
        else:
            prompt = self.system_prompt + content + self.human_prompt
        inputs_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if not self.train_on_inputs:
            labels = [-100] * len(inputs_ids)
        else:
            labels = inputs_ids.copy()
        return {"input_ids": inputs_ids, "labels": labels}
        




def load_tokenized_conversation_dataset(    
    prompter: ConversationPrompter,
    dataset_paths: list[str],
    val_set_size: int,
    select_samples: None | list = None,
    group_by_length: bool = False,
):
    data = load_conversation_dataset_from_paths(dataset_paths)
    if len(data) == 0:
        raise AssertionError(f'Empty dataset with sample number 0. Please check the dataset paths: {dataset_paths}')

    if val_set_size > 0:
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (
            train_val["train"]
            .map(
                prompter.generate_and_tokenize_prompt_mask_input,
                num_proc=mp.cpu_count() - 2,
                remove_columns=data.column_names,
            )
            .filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
        )
        val_data = (
            train_val["test"]
            .map(prompter.generate_and_tokenize_prompt_mask_input, remove_columns=data.column_names)
            .filter(lambda x: len(x["input_ids"]) > 0)
        )
    else:
        # for testing
        if select_samples is not None and len(select_samples) > 0:
            train_data = data.select(select_samples).map(
                prompter.generate_and_tokenize_prompt_mask_input, remove_columns=data.column_names
            ).filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
        else:
            train_data = data.map(
                prompter.generate_and_tokenize_prompt_mask_input,
                num_proc=mp.cpu_count() - 2,
                remove_columns=data.column_names,
            ).filter(lambda x: len(x["input_ids"]) > 0, num_proc=mp.cpu_count() - 2)
            train_data = train_data.shuffle() if not group_by_length else train_data
        val_data = None

    if group_by_length:
        def add_len(example):
            example['ids_len'] = len(example['input_ids'])
            return example
        train_data = train_data.map(add_len, num_proc=mp.cpu_count() - 2, desc='SORT BY LEN').sort('ids_len').remove_columns('ids_len')
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
        lambda example: len(example["conversations"])> 1 and example["conversations"][0].get("from", example["conversations"][0].get("role", "")).lower() in ["human", "user"] ,
        num_proc=mp.cpu_count() -1,
        desc="Remove empty conversations",
    )
    return raw_datasets



PROMPTER_MAP = {
    "mistral": MistralPrompter,
    "llama": ConversationPrompter,
}

def get_prompter(prompter_name, tokenizer, train_on_inputs, cutoff_len, complete_alpha):
    if prompter_name not in PROMPTER_MAP:
        raise ValueError(f"Prompter {prompter_name} not supported")
    return PROMPTER_MAP[prompter_name](tokenizer=tokenizer, train_on_inputs=train_on_inputs, cutoff_len=cutoff_len, complete_alpha=complete_alpha)