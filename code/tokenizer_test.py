# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import logging
import gradio as gr
from peft import PeftModel
import json
import re
import fire
import string
import transformers
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence
import json

from copy import deepcopy
import copy
from torch.utils.data import DataLoader
import datasets
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import DataCollatorForLanguageModeling
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, BloomTokenizerFast, AutoTokenizer
from transformers.models.llama import convert_llama_weights_to_hf
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)
from tokenizer_conversations import *


def print_sample(batch_examples, tokenizer: transformers.PreTrainedTokenizer, skip_s=False):
    """
    1. if label endswith eos token 
    2. input_ids vs label position match
    3. label mask position
    4. label truction by max_length
    4. attention mask 
    Args:
        batch_examples (_type_): _description_
        tokenizer (transformers.PreTrainedTokenizer): _description_
        skip_s (bool, optional): _description_. Defaults to False.
    """
    for i, (input_ids, label, mask) in enumerate(zip(batch_examples["input_ids"], batch_examples["labels"], batch_examples["attention_mask"])):
        # attention mask pad ，label mask input and pad
        input_ids_no_pad = torch.masked_select(input_ids, input_ids != tokenizer.pad_token_id)
        input_ids_no_mask = torch.masked_select(input_ids, mask == 1)
        input_ids_masked = torch.masked_select(input_ids, mask == 0)
        masked_one = torch.masked_select(mask, mask == 1)
        masked_zero = torch.masked_select(mask, mask == 0)
        label_masked = torch.masked_select(label, label == -100)
        label_no_mask = torch.masked_select(label, label != -100)
        input_ids_masked_by_label = torch.masked_select(input_ids, label == -100)
        input_ids_no_masked_by_label = torch.masked_select(input_ids, label != -100)


        print_info = {
            "input_ids": (input_ids, len(input_ids)),
            "input_ids_no_pad": (input_ids_no_pad, len(input_ids_no_pad)),
            "input_ids_no_mask": (input_ids_no_mask, len(input_ids_no_mask)),
            "input_ids_masked": (input_ids_masked, len(input_ids_masked)),
            "input_ids_masked_by_label (expect prompts + pad)": (input_ids_masked_by_label, len(input_ids_masked_by_label)),
            "input_ids_no_masked_by_label (expect label)": (input_ids_no_masked_by_label, len(input_ids_no_masked_by_label)),
            "label": (label, len(label)),
            "label_masked": (label_masked, len(label_masked)),
            "label_no_mask": (label_no_mask, len(label_no_mask)),
            "mask": (mask, len(mask)),
            "masked_zero": (masked_zero, len(masked_zero)),
            "masked_one": (masked_one, len(masked_one)),
        }

        for key, value in print_info.items():
            print(f"{key}: len:{value[1]}\n{value[0]}")
            print("-" * 40)

        print("Decoded Texts:")
        decoded_texts = {
            "input_ids (expect full text with pad)": tokenizer.decode(input_ids, skip_special_tokens=skip_s),
            "input_ids_no_pad (expect full text without pad)": tokenizer.decode(input_ids_no_pad, skip_special_tokens=skip_s),
            "input_ids_no_mask (expect full text without attention_mask)": tokenizer.decode(input_ids_no_mask, skip_special_tokens=skip_s),
            "input_text_masked (expect masked tokens with attention_mask)": tokenizer.decode(input_ids_masked, skip_special_tokens=skip_s),
            "label_no_mask (expect label)": tokenizer.decode(label_no_mask, skip_special_tokens=skip_s),
            "input_ids_masked_by_label (expect masked ids)": tokenizer.decode(input_ids_masked_by_label, skip_special_tokens=skip_s),
            "input_ids_no_masked_by_label (expect label)": tokenizer.decode(input_ids_no_masked_by_label, skip_special_tokens=skip_s),
        }

        for key, value in decoded_texts.items():
            print(f"{key}:\n{value}")
            print("-" * 40)

        print("=" * 40 + 'END' + '=' * 40 + '\n')
    

def print_special_token(tokenizer_hf: transformers.PreTrainedTokenizer):
    print(f"""tokenizer:\n 
          vocab_size:{len(tokenizer_hf)},
          eos:{tokenizer_hf.eos_token},{tokenizer_hf.eos_token_id},
          bos:{tokenizer_hf.bos_token},{tokenizer_hf.bos_token_id},
          pad:{tokenizer_hf.pad_token},{tokenizer_hf.pad_token_id},
          unk:{tokenizer_hf.unk_token},{tokenizer_hf.unk_token_id},
          mask:{tokenizer_hf.mask_token},{tokenizer_hf.mask_token_id},
          cls:{tokenizer_hf.cls_token},{tokenizer_hf.cls_token_id},
          sep:{tokenizer_hf.sep_token},{tokenizer_hf.sep_token_id},
          all_special:{tokenizer_hf.all_special_tokens},{tokenizer_hf.all_special_ids},
          """)



def test_tokenizer(
    dataset_paths: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    cutoff_len: int,
    train_on_inputs: bool,
    sample_ids:list,
    prompt_template_file_name:str
):
    old_len = len(tokenizer)
    prebuild_tokenizer(tokenizer)
    # train_data, val_data = load_tokenized_dataset_alpaca(tokenizer=tokenizer, 
    #                                                   dataset_paths=dataset_paths, 
    #                                                   val_set_size=0, 
    #                                                   template_file=prompt_template_file_name,
    #                                                   cutoff_len=cutoff_len,
    #                                                   train_on_inputs=train_on_inputs, 
    #                                                   select_samples=sample_ids)

    train_data, val_data = load_tokenized_conversation_dataset(
        tokenizer,
        dataset_paths,
        val_set_size=0,
        cutoff_len=cutoff_len,
        train_on_inputs=train_on_inputs,
        select_samples=sample_ids
    )
    print('finish preprocessing')
    new_len = len(tokenizer)
    if len(train_data) < 10:
        data_collator = transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=4, return_tensors="pt", padding=True
            )
        data_loader = DataLoader(train_data, collate_fn=data_collator, batch_size=2)
        data_loader_iter = iter(data_loader)
        for i in range(len(data_loader_iter)):
            batch = next(data_loader_iter)
            print_sample(batch, tokenizer)
        print_special_token(tokenizer)
    def add_len(example):
        example['len'] = len(example['input_ids'])
        return example
    train_data = train_data.map(add_len, num_proc=mp.cpu_count() - 1)
    inputs_ids_len = train_data['len']
    print(f'inputs ids mean len: {np.mean(inputs_ids_len)}, max len: {max(inputs_ids_len)}, min len: {min(inputs_ids_len)}')
    print(f'old: {old_len}, new:{new_len}')
    print(f'dataset train: {len(train_data)}')


def main(
    dataset_paths= list[str],
    base_model: str = "/data/zhangchong/llm_models/llama-7b-hf",
    prompt_template_file_name: str = "./templates/alpaca_short.json",
    cut_off_len: int = 100,
    train_on_inputs: bool = False,
    sample_ids: list = [1230, 2394, 2332, 9]
    
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    kargs = {
        "dataset_paths": dataset_paths, 
        "tokenizer": tokenizer, 
        "cutoff_len": cut_off_len,
        "train_on_inputs": train_on_inputs, 
        "sample_ids": sample_ids,
        "prompt_template_file_name": prompt_template_file_name
        }
    print(kargs)
    test_tokenizer(**kargs)


if __name__ == "__main__":
    fire.Fire(main)




