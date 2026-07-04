import logging
import os
import re
from typing import Any, Union

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _str_to_num(text):
    """Convert text to number, handling various formats and cleaning characters"""
    if text is None:
        return "n/a"

    if isinstance(text, (int, float)):
        return float(text)

    text = str(text)
    text = text.replace("$", "").replace(",", "").replace("_", "")

    # Handle percentages
    if "%" in text:
        text = text.replace("%", "")
        try:
            return float(text) / 100
        except ValueError:
            return "n/a"

    # Handle regular numbers
    try:
        return float(text)
    except ValueError:
        return "n/a"


def _extract_answer_from_response(model_answer):
    """Extract final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""

    # Case 1: <answer>...</answer> tag
    answer_tag_pattern = re.search(r"<answer>(.*?)</answer>", model_answer, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        return answer_content

    # Case 2: find all "Answer: ..." and return the last one
    answer_matches = re.findall(r"Answer:\s*(.*?)(?=(?:\n|$))", model_answer, re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()

    # Case 3: fallback to last line
    lines = model_answer.strip().split("\n")
    return lines[-1].strip()


def _normalize_answer(s):
    """Normalize answer text, remove punctuation and extra whitespace"""
    if s is None:
        return ""

    # If it's a number, return original format
    if isinstance(s, (int, float)):
        return str(s)

    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    if prediction is None or ground_truth is None:
        return 0

    # Handle numeric type exact matching
    pred_num = _str_to_num(prediction)
    truth_num = _str_to_num(ground_truth)

    if pred_num != "n/a" and truth_num != "n/a":
        # Use relative tolerance for comparison
        if isinstance(pred_num, (int, float)) and isinstance(truth_num, (int, float)):
            # Handle special case of zero value
            if truth_num == 0:
                return 1.0 if abs(pred_num) < 1e-5 else 0.0

            # Use relative error
            relative_diff = abs(pred_num - truth_num) / max(abs(truth_num), 1e-10)
            return 1.0 if relative_diff < 1e-5 else 0.0

    # Handle text type exact matching
    return (
        1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0
    )


def reward_multihiertt_exact_match(chat_history_str: str, ground_truth: Any) -> RewardInfo:
    pred = _extract_answer_from_response(chat_history_str)
    if not pred:
      return RewardInfo(reward=0.0, completed=0.0)
    
    em = _exact_match_score(pred, ground_truth)
    return RewardInfo(reward=float(em), completed=1.0)
