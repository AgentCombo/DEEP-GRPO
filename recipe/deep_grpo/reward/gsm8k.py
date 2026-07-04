# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging

from recipe.deep_grpo.protocol import RewardInfo


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_ASSISTANT_START_MARKERS = (
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>",
    "[/INST]",
    "### Assistant:",
    "Assistant:",
)
_ASSISTANT_END_MARKERS = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
)
_HASH_FINAL_ANSWER_RE = re.compile(
    r"####\s*(?:\$?\s*)?(?:\\boxed\b|[-+]?\d)"
)


def _extract_model_response(solution_str: str) -> str:
    """Best-effort extraction of the assistant response from chat-formatted text."""
    best_idx = -1
    response = solution_str
    for marker in _ASSISTANT_START_MARKERS:
        idx = solution_str.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            response = solution_str[idx + len(marker):]

    response = response.lstrip()
    end_positions = [
        idx for marker in _ASSISTANT_END_MARKERS
        if (idx := response.find(marker)) >= 0
    ]
    if end_positions:
        response = response[:min(end_positions)]
    return response


def has_repeated_final_answer_markers(solution_str: str) -> bool:
    response = _extract_model_response(solution_str)
    hash_answer_count = len(_HASH_FINAL_ANSWER_RE.findall(response))
    boxed_count = response.count("\\boxed")
    return hash_answer_count > 1 or boxed_count > 1


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(chat_history_str, ground_truth, method="flexible", format_score=0.0, score=1.0) -> RewardInfo:
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if has_repeated_final_answer_markers(chat_history_str):
        return RewardInfo(reward=0, completed=0)

    answer = extract_solution(solution_str=chat_history_str, method=method)
    if answer is None:
        return RewardInfo(reward=0,
                          completed=0)
    else:
        if answer == ground_truth:
            return RewardInfo(reward=score,
                              completed=1)
        else:
            return RewardInfo(reward=format_score,
                              completed=1)
