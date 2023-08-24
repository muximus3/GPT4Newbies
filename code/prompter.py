# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import logging
import os.path as osp
from typing import Union
import random
import json
sys.path.append(os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/..'))
logger = logging.getLogger(__name__)

class AlpacaPrompter(object):
    __slots__ = ("template", "_verbose", "system", "desc", "default_role")

    def __init__(self, template_file_path: str = "", verbose: bool = False):
        """Strongly relay on a template with format like: 
        {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{role}:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n{role}:\n",
        "response_split": "{role}:",    
        "default_role": "### Response"
        }
        Args:
            template_file_name (str, optional): temple file path. Defaults to "".
            verbose (bool, optional): if set True, prompter would log every prompt and response. Defaults to False.

        Raises:
            ValueError: _description_
        """
        self._verbose = verbose
        if not osp.exists(template_file_path):
            raise ValueError(f"Can't read {template_file_path}")
        with open(template_file_path) as fp:
            self.template = json.load(fp)
            self.desc = self.template['description']
            self.default_role = self.template['default_role']
        if self._verbose:
            logger.info(
                f"Using prompt template {template_file_path}: {self.desc}, role:{self.default_role}"
            )
    def user_prompt(
        self,
        instruction: str,
        input_ctx: str="",
        role: str="",
        system: str="",
    ):
        # returns the user prompt from instruction and optional input
        template_prompt = self.template["prompt_input"] if input_ctx else self.template["prompt_no_input"]
        if not role: 
            role = self.default_role
        if input_ctx:
            prompt = template_prompt.format(
                    instruction=instruction, input=input_ctx, role=role
                )
        else:
            prompt = template_prompt.format(
                    instruction=instruction, role=role
                )
        if system:
            prompt = f"{self.template['system']}{system}{prompt}"
        prompt = prompt.lstrip()  
        if self._verbose:
            logger.info(f'user prompt:{prompt}')
        return prompt

    def format_response(self, output: str, role: str="") -> str:
        # potential bug: response_split is too common for a random sequence
        if not role:
            role = self.default_role
        response_split  = self.template["response_split"].format(role=role)
        res = output
        res_splits = output.rsplit(response_split, maxsplit=1)
        if len(res_splits) > 1:
            res = res_splits[1].strip()
            return res if res else output
        if self._verbose:
            print(res)
        return res


            
# if __name__ == "__main__":
#     p = AlpacaPrompter('./templates/conversitions.json')
#     print(p.user_prompt('hello follow', 'a + b', role='AI', system='Now you are a bad AI'))
#     print('-' * 20)
#     print(p.user_prompt('hello follow', 'a + b', role='AI', system=''))
#     print('-' * 20)
#     print(p.user_prompt('hello follow', 'a + b', role=''))
#     print('-' * 20)
#     print(p.format_response('hello coder'))
#     print('-' * 20)
#     print(p.format_response('hello helpful coder Assistant: sdf: a helpful ### Re AI助手:sponse: f: coder sdfs helpful coder: ', role=''))