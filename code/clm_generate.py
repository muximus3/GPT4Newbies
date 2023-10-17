# -*- coding: utf-8 -*-
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import os
import sys
import numpy as np
import torch
import logging
import gradio as gr
from peft import PeftModel
import time
import json
import fcntl
import fire
import transformers
from tqdm import tqdm
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
sys.path.append(os.path.normpath(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from prompter import AlpacaPrompter
from data_utils import  get_left_data, df_reader
logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
    test_data_path: str = None,
    load_8bit: bool = False,
    model_name_or_path: str = "",
    tokenizer_name_or_path: str = "",
    state_dict_path: str = "",
    lora_weights: str = "",
    test_output_file="",
    lora=False,
    # Allows to listen on all interfaces by providing '0.
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
    template_file: str = None,
):
    assert model_name_or_path, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    if lora:
        assert os.path.isdir(
            lora_weights), ("Please specify a --lora_weights, e.g. --lora_weights='tloen/alpaca-lora-7b'")
        print(
            f'base model=========>: {model_name_or_path}\nlora weights==========>: {lora_weights}')


    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    if not template_file:
        raise AssertionError(f'Please specify a template file for Prompter')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if device == "cuda":
        print('cuda')
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                # device_map={'': 0} # fix AttributeError: 'NoneType' object has no attribute 'device'
            )
    elif device == "mps":
        print('mps')
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # deepspeed state dict file 
    if not lora and os.path.isfile(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path))
    prompter = AlpacaPrompter(template_file)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def evaluate(
        data_point: dict,
        temperature=0.01,
        top_p=0.82,
        top_k=10,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        instruction = data_point["instruction"]
        input = data_point.get("input", "")
        role = data_point.get("role", "")
        system = data_point.get("system_prompt", data_point.get("system", ""))
        # if system in instruction:
        instruction = instruction.replace(system, '').strip()
        prompt = prompter.user_prompt(instruction=instruction, input_ctx=input, role=role, system=system)
        # print(prompt)
        features = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = features['input_ids'].to("cuda")
        attention_mask = features['attention_mask'].to("cuda")
        generation_config = GenerationConfig(
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            # do_sample=True,
            # num_beams=num_beams,
            # min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            # repetition_penalty=1.2,
            # no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # length_penalty=0.6,
            # early_stopping=True,
            **kwargs,
        )
        with torch.inference_mode():
            model.eval()
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(
            s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print('-'*40)
        print(f'raw_output:\n{output}')
        format_out = prompter.format_response(output, role=data_point.get("role", ""))
        model_name = lora_weights if lora else os.path.basename(state_dict_path)
        model_name = model_name if model_name else model_name_or_path
        if test_output_file:
            data_point['output'] = format_out 
            data_point['model'] = model_name
            data_point['generation_setting'] = generation_config.to_diff_dict()
            fcntl.flock(writer, fcntl.LOCK_EX)
            try:
                writer.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                writer.flush()
                time.sleep(0.1)
            finally:
                fcntl.flock(writer, fcntl.LOCK_UN)
        return format_out

    if test_data_path and os.path.isfile(test_data_path):
        total_msgs = df_reader(test_data_path).fillna('').to_dict(orient='records')
        if not os.path.isfile(test_output_file):
            left_msgs = total_msgs
        else:
            processed_msgs = df_reader(test_output_file).fillna('').to_dict(orient='records')
            if len(processed_msgs) > 0:
                left_msgs = get_left_data(total_msgs, processed_msgs)
            else:
                left_msgs = total_msgs
        if test_output_file:
            writer = open(test_output_file, "a+")
        for item in left_msgs:
            evaluate(item)
    else:
        def evaluate_single(
            instruction: str,
            input_str: str = '',
            role: str = 'Assistant',
            temperature=0.8,
            top_p=1,
            top_k=40,
            num_beams=4,
            max_new_tokens=1024
        ):
            data_point = {'instruction': instruction, 'input': input_str,
                          'output': '', 'target': '', 'role': role}
            return evaluate(data_point, temperature, top_p, top_k, num_beams, max_new_tokens)
        gr.Interface(
            fn=evaluate_single,
            inputs=[
                gr.components.Textbox(
                    lines=2, label="Instruction", placeholder="Tell me about alpacas."
                ),
                gr.components.Textbox(
                    lines=2, label="Input", placeholder="none"),
                gr.components.Textbox(
                    lines=2, label="role", placeholder="Assistant"),
                gr.components.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature"),
                gr.components.Slider(minimum=0, maximum=1,
                                     value=0.75, label="Top p"),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=40, label="Top k"
                ),
                gr.components.Slider(minimum=1, maximum=4,
                                     step=1, value=4, label="Beams"),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="🦙🌲 Alpaca-LoRA-XiaoDuo-v0.1",
            description=f"基于 7B LLaMA 模型:{lora_weights}",
        ).queue().launch(server_name=server_name, share=share_gradio)




if __name__ == "__main__":
    fire.Fire(main)

