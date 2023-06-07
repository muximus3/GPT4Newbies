# -*- coding: utf-8 -*-
import os
import sys
import logging
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
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim, set_seed
from accelerate.state import AcceleratorState
from deepspeed.accelerator import get_accelerator
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from tqdm import tqdm
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, LlamaForCausalLM, LlamaTokenizer
from pydantic import BaseModel
from typing import Union, List
import math
import fire
import wandb

sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)
from tokenizer_utils import prebuild_tokenizer, load_tokenized_dataset

torch.backends.cuda.matmul.allow_tf32 = True


class TrainArgs(BaseModel):
    # model/tokenizer
    model_name: str
    tokenizer_name: str
    dataset_paths: List[str]
    checkpoint: Union[str, None] = None
    output_dir: str
    prompt_template_path: str
    save_name: str = None
    streaming: bool = False
    train_on_inputs = True
    gradient_checkpointing: bool = True

    num_epochs: int = 4
    max_length: int = 1024
    micro_batch_size: int = 1
    num_proc: int = 64

    lr: float = 2e-5
    min_lr: float = 0.0
    weight_decay: float = 0.0
    eval_every: int = 200
    print_loss_every: int = 50
    save_every: int = 800
    log_grads_every: int = 400
    warmup_steps: int = 100
    max_eval_num: int = 20

    lora: bool = False
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_r: int = 8
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    seed: int = 42

    wandb: bool = True
    wandb_entity: str = "gpt4newbies"
    wandb_project_name: str = "huggingface"


def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log


def evaluate(model, val_dataloader, accelerator):
    model.eval()
    val_loss = MeanMetric(nan_strategy="error").to(model.device)

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss = model(**batch).loss

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

            val_loss.update(loss_values["loss"])

    return val_loss


def train(accelerator, config: TrainArgs):
    set_seed(config.seed)
    accelerator.free_memory()
    accelerator.print(config)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = LlamaTokenizer.from_pretrained(config.tokenizer_name)

    checkpoint = config.gradient_checkpointing
    # A device map needs to be passed to run convert models into mixed-int8 format. Please run`.from_pretrained` with `device_map='auto'
    model = LlamaForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False if checkpoint else True,
        torch_dtype=torch.float16,
        # device_map="auto",
    )

    prebuild_tokenizer(tokenizer, model)
    with accelerator.main_process_first():
        train_dataset, val_dataset = load_tokenized_dataset(
            tokenizer,
            config.dataset_paths,
            config.max_eval_num,
            config.prompt_template_path,
            config.max_length,
            config.train_on_inputs,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config.micro_batch_size,
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=config.micro_batch_size,
    )
    accelerator.wait_for_everyone()

    if checkpoint:
        model.gradient_checkpointing_enable()

    if config.lora:
        target_modules = (
            config.lora_target_modules if config.lora_target_modules else None
        )
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Accelerate official example
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    # karpathy doesn't decay embeddding, maybe we should exclude
    # https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1#diff-2075fa9c224b395be5bda85544dd36572b59c76c54562819eadadbf268602834R157s
    accelerator.print(f"Optimizer class: {optimizer_cls.__name__}")
    optimizer = optimizer_cls(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )

    # decay to min_lr instead of 0
    lr_ratio = config.min_lr / config.lr
    dataset_size = len(train_dataloader) * config.micro_batch_size
    total_batch_size = (
        config.micro_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(dataset_size / (config.micro_batch_size * accelerator.num_processes))
    total_num_steps = steps_per_epoch * config.num_epochs
    # instead of decaying to zero, decay to ratio of min_lr / lr
    total_num_steps += int(total_num_steps * lr_ratio) + config.warmup_steps
    # train_batch_size equal to micro_batch_per_gpu * gradient_acc_step * world_size
    accelerator.print(f"Accelerate state:\n\n{AcceleratorState()}\n")
    accelerator.print(f"Train dataset size: {dataset_size}")
    accelerator.print(
        f"Steps per epoch: {steps_per_epoch}\nTrain epochs:{config.num_epochs}\nTotal training steps: {total_num_steps}"
    )
    accelerator.print(
        f"Total batch size: {total_batch_size}\nGradient_acc_steps:{gradient_accumulation_steps}"
    )

    # Using Accelerate Deepspeed PLugin
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        accelerator.print("linear schedule")
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps * accelerator.num_processes,
            num_training_steps=total_num_steps,
        )
        # scheduler = get_scheduler(
        #     name=config.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=config.warmup_steps * accelerator.num_processes,
        #     num_training_steps=total_num_steps,
        # )
        # accelerator.print("cosine schedule")
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=config.warmup_steps * accelerator.num_processes,
        #     num_training_steps=total_num_steps,
        # )
    else:
        # Using deepspeed with config file
        accelerator.print("dummy schedule")
        scheduler = DummyScheduler(
            optimizer,
            total_num_steps=config.warmup_steps,
            warmup_num_steps=config.warmup_steps,
        )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # setup for saving training states in case preemption
    accelerator.register_for_checkpointing(scheduler)

    if config.checkpoint:
        accelerator.load_state(config.checkpoint)
        accelerator.print(f"Resumed from checkpoint: {config.checkpoint}")
        path = os.path.basename(config.checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("step_", ""))
        accelerator.skip_first_batches(train_dataloader, resume_step)
        accelerator.print(f"Resuming from step {resume_step}")

    # log gradients
    if accelerator.is_main_process and config.wandb:
        wandb.watch(model, log_freq=config.log_grads_every, log="all")

    for epoch in range(config.num_epochs):
        train_loss = MeanMetric(nan_strategy="error").to(model.device)
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss

            # gather loss before backprop in case of gradient accumulation
            loss_values = accelerator.gather_for_metrics(
                {"loss": loss.detach().float()}
            )
            train_loss.update(loss_values["loss"])

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            # get gradient norm of all params

            # log LR in case something weird happens
            if step > 0 and step % (config.eval_every // 10) == 0:
                if config.wandb:
                    curr_step = step + epoch * len(train_dataloader)
                    accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % config.print_loss_every == 0:
                accelerator.print(
                    f"Epoch:{epoch}, step:{step}, loss:{train_loss.compute()}"
                )
            # if step > 0 and step % config.save_every == 0:
            #     curr_step = step + epoch * len(train_dataloader)
            #     accelerator.save_state(f"{config.output_dir}/step_{curr_step}")

            if step > 0 and (
                step % config.eval_every == 0 or step == len(train_dataloader) - 1
            ):
                val_loss = evaluate(model, val_dataloader, accelerator)

                log_train = {"train_loss": train_loss.compute()}
                log_val = {"val_loss": val_loss.compute()}

                if config.wandb:
                    curr_step = step + epoch * len(train_dataloader)
                    accelerator.log({**log_train, **log_val}, step=curr_step)

                accelerator.print(f"Current LR: {scheduler.get_last_lr()[0]}")
                accelerator.print(format_metrics(log_train, "train", f" step {step} "))
                accelerator.print(format_metrics(log_val, "val", f" step {step} "))

                train_loss.reset()
        # in case of warning: pytorch allocator cache flushes since last step. \
        # this happens when there is high memory pressure and is detrimental to performance.
        # if this is happening frequently consider adjusting settings to reduce memory consumption.
        # If you are unable to make the cache flushes go away consider adding get_accelerator().empty_cache() calls in your training loop to ensure that all ranks flush their caches at the same time
        get_accelerator().empty_cache()

        accelerator.print(f"Epoch {epoch} finished")
        accelerator.print(f"Saving checkpoint to:{config.output_dir}")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # try:
        #     if accelerator.is_main_process and config.save_name:
        #         unwrapped_model.push_to_hub(config.save_name + f"-epoch_{epoch}", private=True)
        # except Exception as e:
        #     accelerator.print(e)
        #     accelerator.print(f"Failed to push to hub")

        unwrapped_model.save_pretrained(
            f"{config.output_dir}/epoch_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{config.output_dir}/final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.end_training()


def main(**kargs):
    train_config = TrainArgs(**kargs)
    if train_config.wandb:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name=train_config.wandb_project_name,
            config=train_config,
            init_kwargs={"wandb": {"entity": train_config.wandb_entity}},
        )
    else:
        accelerator = Accelerator()

    train(accelerator=accelerator, config=train_config)


if __name__ == "__main__":
    fire.Fire(main)
