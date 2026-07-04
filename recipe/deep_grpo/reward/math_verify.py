import os
import logging
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from recipe.deep_grpo.protocol import RewardInfo

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


NUM_WORKERS = 32
MP_CTX = multiprocessing.get_context('spawn')
PROCESS_EXECUTOR = ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=MP_CTX)


def _verification_worker(model_output: str, ground_truths: list[str], timeout_score: float) -> float:
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        
        ground_truths_boxed = [f"\\boxed{{{gt}}}" for gt in ground_truths]
        
        ret_score, _ = verify_func(ground_truths_boxed, [model_output])
        return float(ret_score)

    except TimeoutException:
        return timeout_score
    except Exception:
        return 0.0

async def _compute_score(model_output: str, ground_truths: list[str], timeout_score: float = 0.0) -> float:
    loop = asyncio.get_running_loop()
    
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                PROCESS_EXECUTOR,
                _verification_worker,
                model_output,
                ground_truths,
                timeout_score
            ),
            timeout=12.0
        )
        return result

    except asyncio.TimeoutError:
        logger.warning(f"Async reward computation timed out.")
        return timeout_score
        
    except Exception as e:
        logger.error(f"Error in async compute_score: {e}")
        return 0.0
    
async def compute_score(chat_history_str, ground_truth) -> RewardInfo:
    assert ground_truth is not None
    if isinstance(ground_truth, list):
        gt_list = [str(g) for g in ground_truth]
    else:
        gt_list = [str(ground_truth)]
    reward = await _compute_score(chat_history_str, gt_list)
    return RewardInfo(reward=reward, completed=1)