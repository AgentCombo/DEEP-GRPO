# https://github.com/xlang-ai/DS-1000/blob/main/test_ds1000.py
# https://github.com/ruc-datalab/DeepAnalyze/blob/main/playground/DS-1000/run_deepanalyze.py

from typing import List

import re
import logging
import os
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from . import execution

from recipe.deep_grpo.protocol import RewardInfo


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

semaphore = asyncio.Semaphore(128)
ds1000_executor = ThreadPoolExecutor(max_workers=128)

def extract_python_block(text):
    pattern = r"```python.*?```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    pattern_plain = r"```(.*?)```"
    matches = re.findall(pattern_plain, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
        
    return None

def postprocess(code: List | str):
    # whereas Codex002 naturally followed the given format, GPT3.5/GPT4 generates all kinds of tags despite being asked not to
    if type(code) is list:
        code = code[0]
    code = code.split('</code>')[0]
    code = code.replace('```python', '')
    code = code.split('```')[0]
    code = code.split('\nEND SOLUTION')[0]
    code = code.replace('<code>', '')
    return code

def run_ds1000_check(test_program, timeout=120):
    try:
        result = execution.check_correctness(test_program, timeout=timeout)
        return result['passed']
    except Exception as e:
        logger.warning(f"DS1000 execution internal error: {e}")
        return False

async def evaluate_ds1000(chat_history_str: str, 
                          code_context: str,
                          timeout: int = 120) -> RewardInfo:

    async with semaphore:
        extracted_code = extract_python_block(chat_history_str)
        
        if extracted_code is None:
            return RewardInfo(reward=0.0, completed=0)

        clean_code = postprocess(extracted_code)

        test_program = (
            code_context + '\n'
            + f'code = {repr(clean_code)}\n'
            + 'test_execution(code)\n'
            + (
                "test_string(code)\n"
                if "test_string(" in code_context
                else "\n"
            )
        )
        
        loop = asyncio.get_running_loop()
        
        try:
            is_passed = await loop.run_in_executor(
                ds1000_executor,
                functools.partial(run_ds1000_check, test_program, timeout)
            )
            score = 1.0 if is_passed else 0.0
            return RewardInfo(reward=score, completed=1)

        except Exception as e:
            logger.warning(f"Failed to evaluate DS1000 sample. Exception: {e}")
            return RewardInfo(reward=0.0, completed=0)