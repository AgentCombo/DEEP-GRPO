import re
import logging
import os
import sys
import asyncio

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

semaphore = asyncio.Semaphore(128)


def extract_answer(text):
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None


async def evaluate_submission(chat_history_str: str,
                              eval_script_path: str, 
                              submission_file: str, 
                              ground_truth_file: str, 
                              baseline_result: float, 
                              gold_result: float) -> RewardInfo:
    
    # eval submission file, we have cleaned submission file before run val
    # see recipe/deep_grpo/agent_loop/deep_analyze_agent_loop.py _run_val

    async with semaphore:
        answer = extract_answer(chat_history_str)
        if answer is None:
            return RewardInfo(reward=0,
                              completed=0)

        command = [
            sys.executable, 
            eval_script_path,
            "--answer_file", ground_truth_file,
            "--predict_file", submission_file,
            "--path", "dummy",
            "--name", "dummy"
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_message = stderr.decode('utf-8', errors='replace').strip()
                logger.warning(
                    f"Evaluation script for {submission_file} failed with return code {proc.returncode}. "
                    f"Error: {error_message}"
                )
                return RewardInfo(reward=0,
                                  completed=0)

            output = stdout.decode('utf-8', errors='replace').strip()
            model_result = float(output)

        except ValueError:
            logger.warning(f"Could not convert stdout to float. Raw output: '{stdout.decode().strip()}'")
            return RewardInfo(reward=0,
                              completed=0)
        except Exception as e:
            logger.warning(f"Failed to evaluate {submission_file} against {ground_truth_file} using {eval_script_path}, the exception is {e}")
            return RewardInfo(reward=0,
                              completed=0)
        
        larger_better = False
        if gold_result > baseline_result:
            larger_better = True
        
        if larger_better:
            score = max(0, (model_result - baseline_result) / (gold_result - baseline_result))
        else:
            score = max(0, (baseline_result - model_result) / (baseline_result - gold_result))
        
        return RewardInfo(reward=score,
                          completed=1)


        