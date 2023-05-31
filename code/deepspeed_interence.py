# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import logging
import deepspeed
import fire
import torch
from transformers import pipeline
sys.path.append(os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/..'))
logger = logging.getLogger(__name__)

def main(prompt, model_name_or_path, tokenizer_name_or_path, world_size=8):
    print('================>')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', world_size))
    generator = pipeline('text-generation', model=model_name_or_path, tokenizer=tokenizer_name_or_path,
                        device=local_rank)

    generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.bfloat16,
                                            replace_with_kernel_inject=True)

    string = generator(prompt, do_sample=True, min_length=250)
    print(f'================>{string}')
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(string)



if __name__ == "__main__":
    fire.Fire(main)
