# -*- coding: utf-8 -*-
import os
import sys
import logging
import re
import shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_scheduler,
    get_cosine_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_pt_utils import get_parameter_names
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch
import os
import torch
import numpy as np
import multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim, set_seed
from accelerate.state import AcceleratorState
from peft import AutoPeftModelForCausalLM, LoraConfig
from deepspeed.accelerator import get_accelerator
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from transformers import (
    DefaultDataCollator,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from pydantic import BaseModel
from typing import Union, List
import math
import fire
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
    DPOTrainer,
)
from trl.core import respond_to_batch
from trl.trainer.utils import DPODataCollatorWithPadding
import wandb

sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)
from tokenizer_conversations import (
    prebuild_tokenizer,
    load_tokenized_conversation_dataset,
)
from data_utils import load_dataset_from_path


class TrainArgs(BaseModel):
    # model/tokenizer
    model_name_or_path: str
    tokenizer_name: str
    dataset_path: str
    output_dir: str
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-6
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 50
    weight_decay: float = 0.0
    optimizer_type: str = "paged_adamw_32bit"
    # the beta parameter for DPO loss
    beta: float = 0.2

    max_steps: int = 5000
    max_length: int = 1024
    logging_steps: int = 20
    save_steps: int = 500
    eval_steps: int = 100
    max_eval_num: int = 200
    max_prompt_length: int = 512

    lora: bool = True
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_r: int = 8

    report_to: str = "wandb"
    run_name: str = "dpo_llama2"
    ignore_bias_buffers: bool = False

def load_compare_dataset(dataset_path, val_set_size=0):
    # data = {'id': '', 'system_prompt': '', 'dataset_name': '', 'prompt': '', 'chosen': '', 'rejected': ''}
    raw_dataset = load_dataset_from_path(dataset_path)
    column_names = ["id", "system_prompt", "dataset_name"]

    def format_data(sample):
        return {
            "prompt": "System:\n"
            + sample["system_prompt"]
            + "\n\nHuman:\n"
            + sample["prompt"]
            + "\n\nAssistant:\n"
            if sample["system_prompt"] != ""
            else "Human:\n" + sample["prompt"] + "\n\nAssistant:\n",
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    if val_set_size > 0:
        train_val = raw_dataset.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].map(
            format_data,
            num_proc=mp.cpu_count() - 2,
            remove_columns=column_names,
        )
        val_data = train_val["test"].map(format_data, remove_columns=column_names)
    else:
        train_data = raw_dataset.map(
            format_data,
            num_proc=mp.cpu_count() - 2,
            remove_columns=column_names,
        )
        val_data = None
    return train_data, val_data


def train(args: TrainArgs):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # device_map = {"": Accelerator().local_process_index}
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # quantization_config=bnb_config,
        # device_map=device_map
    )
    model.config.use_cache = False
    # model = prepare_model_for_kbit_training(model)
    ref_model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # quantization_config=bnb_config,
        # device_map=device_map
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.unk_token_id

    train_dataset, val_dataset = load_compare_dataset(
        args.dataset_path, args.max_eval_num
    )
    print(f"Train dataset: {len(train_dataset)}")
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.run_name,
    )

    peft_config = (
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        if args.lora
        else None
    )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        # data_collator=DPODataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=args.max_length,                                                  
        #                                          max_prompt_length=args.max_prompt_length, 
        #                                          label_pad_token_id=-100, 
        #                                          padding_value=0,
        #                                          truncation_mode='keep_end')
    )
    # dpo_trainer.compute_metrics = compute_metrics
    # 6. train
    # dpo_trainer.ref_model = Accelerator().prepare(dpo_trainer.ref_model)
    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    # 7. save
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)


def main(**kargs):
    train_config = TrainArgs(**kargs)
    train(args=train_config)


if __name__ == "__main__":
    # fire.Fire(main)
    dataset_path = (
        "/data/zhangchong/train_data/open_domain_prompt_zh/compareset_13w.jsonl"
    )
    train_dataset, val_dataset = load_compare_dataset(dataset_path, 200)
    select = train_dataset.shuffle(seed=42).select(range(10))
    item = next(iter(select))
    print(item["prompt"])
    print('--------------------------------------')
    print(item["chosen"])
    print('--------------------------------------')
    print(item["rejected"])
