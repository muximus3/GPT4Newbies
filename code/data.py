import json
import argparse
import os
import glob
from datasets import load_dataset
from transformers import AutoTokenizer

# todo more type of data/load function
def load_dataset_and_tokenize(tokenizer, dataset_name_or_path, train_val_split_ratio=0.05):
    if os.path.isdir(dataset_name_or_path):
        train_files = glob.glob(os.path.join(dataset_name_or_path, "*train*"))
        val_files = glob.glob(os.path.join(dataset_name_or_path, "*val*"))
        # test_files = glob.glob(os.path.join(dataset_name_or_path, "*test*"))
        extension = train_files[0].split(".")[-1]
        if len(val_files) == 0:
            raw_dataset = load_dataset(extension, data_files=train_files, split="train")
        else:
            raw_dataset = load_dataset(extension, data_files={"train": train_files, "validition": val_files})
            
    elif os.path.isfile(dataset_name_or_path):
        extension = dataset_name_or_path[0].split(".")[-1]
        raw_dataset = load_dataset(extension, data_files=dataset_name_or_path, split="train")
    
    else:
        # todo downloading dataset
        raw_dataset = load_dataset(name=dataset_name_or_path)
        train_dataset = raw_dataset["train"]
        validation_dataset = raw_dataset["validation"]
    
        




        




    