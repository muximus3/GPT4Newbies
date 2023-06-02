# -*- coding: utf-8 -*-
import os
import sys
import logging
from datasets import load_dataset, concatenate_datasets
import transformers
import pandas as pd
import multiprocessing as mp
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
from prompter import Prompter

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


def load_tokenized_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_paths: list[str],
    val_set_size: int,
    template_file: str,
    cutoff_len: int = 512,
    train_on_inputs: bool = False,
    select_samples: None | list = None,
):
    prompter = Prompter(template_file)

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if len(result["input_ids"]) >= cutoff_len:
            result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][cutoff_len - 1] = 1

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        # relay on a specify data format
        instruction, input_ctx, output, role = (
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
            data_point.get("role"),
        )

        user_prompt = prompter.user_prompt(
            instruction=instruction, input_ctx=input_ctx, role=role
        )
        # nip in the bud
        if len(user_prompt) >= cutoff_len // 2 and not train_on_inputs:
            new_len = min(len(user_prompt) // 2, cutoff_len // 2)
            # Since we have truncated the tail, we need to add '\n' in case the BPE tokenizer generates unexpected results.
            user_prompt = user_prompt[:new_len] + "\n"

        full_prompt = f"{user_prompt}{output}"
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            # If the last word of user prompt is whitespace, the tokenized result may be different from the full prompt with BPE tokenizing.
            # Label masking based on tokenized_user_prompt may produce unexcepted result
            tokenized_user_prompt = tokenizer.encode(
                user_prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            tokenized_user_prompt_len = len(tokenized_user_prompt)
            if tokenized_user_prompt_len >= len(tokenized_full_prompt["input_ids"]):
                print(
                    f'smt wrong, user prompt len: {len(user_prompt)}, token:{tokenized_user_prompt_len}, full prompt len:{len(full_prompt)}, token: {len(tokenized_full_prompt["input_ids"])}'
                )
            tokenized_full_prompt["labels"] = [
                IGNORE_INDEX
            ] * tokenized_user_prompt_len + tokenized_full_prompt["labels"][
                tokenized_user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    data = load_dataset_from_paths(dataset_paths=dataset_paths)
    if len(data) == 0:
        raise AssertionError(f'Empty dataset with sample number 0. Please check the dataset paths: {dataset_paths}')

    if val_set_size > 0:
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (
            train_val["train"]
            .shuffle()
            .map(
                generate_and_tokenize_prompt,
                num_proc=min(mp.cpu_count() - 1, 16),
                remove_columns=data.column_names,
            )
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data.column_names)
        )
    else:
        # for testing
        if select_samples is not None and len(select_samples) > 0:
            train_data = data.select(select_samples).map(
                generate_and_tokenize_prompt, remove_columns=data.column_names
            )
        else:
            train_data = data.shuffle().map(
                generate_and_tokenize_prompt,
                num_proc=min(mp.cpu_count() - 1, 16),
                remove_columns=data.column_names,
            )
        val_data = None

    return train_data, val_data


def load_dataset_from_paths(
    dataset_paths: list[str],
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """load multiple datasets(with/without the same columns) into DatasetDict/Dataset.
    Json file should include these columns:
    ['instruction', 'input', 'output']
    or [instruction', 'context', 'response']
    or ['kind', 'input', 'target']
    or ['input', 'target']
    or ['prompt','response']

    Args:
        dataset_paths (List[str]): List of datasets' paths.

    Raises:
        KeyError:

    Returns:
        Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]: _description_
    """
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
        if len({"instruction", "input", "output"} - set(raw_dataset.column_names)) == 0:
            pass
        # dolly data format
        elif (
            len({"instruction", "context", "response"} - set(raw_dataset.column_names))
            == 0
        ):
            raw_dataset = raw_dataset.rename_columns(
                {"context": "input", "response": "output"}
            )
        # firefly data format
        elif len({"kind", "input", "target"} - set(raw_dataset.column_names)) == 0:
            raw_dataset = raw_dataset.rename_columns(
                {"input": "instruction", "target": "output", "kind": "category"}
            )
            raw_dataset = raw_dataset.add_column("input", [""] * len(raw_dataset))
        # belle data format
        elif len({"input", "target"} - set(raw_dataset.column_names)) == 0:
            raw_dataset = raw_dataset.rename_columns(
                {"input": "instruction", "target": "output"}
            )
            raw_dataset = raw_dataset.add_column("input", [""] * len(raw_dataset))
        # gpt4all data format
        elif len({"prompt", "response"} - set(raw_dataset.column_names)) == 0:
            raw_dataset = raw_dataset.rename_columns(
                {"prompt": "instruction", "response": "output"}
            )
            raw_dataset = raw_dataset.add_column("input", [""] * len(raw_dataset))
        else:
            raise KeyError(
                f"Data format error: {dataset_path}, columns: {raw_dataset.column_names}"
            )
        raw_datasets_list.append(raw_dataset)

    if len(raw_datasets_list) == 1:
        raw_datasets = raw_datasets_list[0]
    else:
        raw_datasets = concatenate_datasets(raw_datasets_list)
    assert len({"instruction", "input", "output"} - set(raw_datasets.column_names)) == 0, raw_datasets.column_names
    raw_datasets = raw_datasets.filter(
        lambda example: example["instruction"] != "" and len(example["output"]) > 0,
        num_proc=min(mp.cpu_count() - 1, 16),
        desc="Remove empty instruction and output",
    )
    return raw_datasets


