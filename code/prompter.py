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
    __slots__ = ("template", "_verbose")

    def __init__(self, template_file_path: str = "", verbose: bool = False):
        self._verbose = verbose
        if not osp.exists(template_file_path):
            raise ValueError(f"Can't read {template_file_path}")
        with open(template_file_path) as fp:
            self.template = json.load(fp)
            desc = self.template['description']
        if self._verbose:
            logger.info(
                f"Using prompt template {template_file_path}: {desc}"
            )
    def user_prompt( self, instruction: str, system: str = ""):
        # returns the user prompt from instruction and optional input
        template_prompt = self.template["prompt"] if system else self.template["prompt_no_system"]
        if system:
            prompt = template_prompt.format(system=system, instruction=instruction)
        else:
            prompt = template_prompt.format(instruction=instruction)
        prompt = prompt.lstrip()  
        if self._verbose:
            logger.info(f'user prompt:{prompt}')
        return prompt

    def format_response(self, output: str, prompt: str) -> str:
        # potential bug: response_split is too common for a random sequence
        return output[len(prompt):]


            
if __name__ == "__main__":
    p = AlpacaPrompter('./templates/mistral.json')
    print(p.user_prompt('hello follow', system='Now you are a bad AI'))
    print('-' * 20)
    print(p.user_prompt('hello follow',  system=''))
    print('-' * 20)
    print(p.user_prompt('hello follow', 'a + b'))
    print('-' * 20)
    print(p.format_response('hello coder', ''))
    print('-' * 20)
    print(p.format_response(output='hello helpful coder Assistant: sdf: a helpful ### Re AI助手:sponse: f: coder sdfs helpful coder: ', prompt='hello helpful coder Assistant:'))