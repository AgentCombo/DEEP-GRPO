import logging
import os
import re
import string
from typing import Any

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_answer_from_response(model_answer):
    """
    Extract final answer from the model response, supporting multiple formats:
    1. "Answer: xxx" format
    2. "<answer>xxx</answer>" tag format
    3. "<answer>Answer: xxx</answer>" combined format
    """
    # If response is empty, return empty string
    if not model_answer:
        return ""

    # Check if there's an <answer> tag
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)
    if answer_tag_pattern:
        # Extract content from <answer> tag
        answer_content = answer_tag_pattern.group(1).strip()

        # Check if there's an "Answer:" mark inside the tag
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        # Otherwise return the full content inside the tag
        return answer_content

    # If no <answer> tag but there's an "Answer:" mark
    elif "Answer:" in model_answer:
        return model_answer.split("Answer:")[-1].strip()

    # If no markers, return the entire response
    return model_answer.strip()

def _clean_answer_tags(answer_text):
    if not answer_text:
        return ""

    # Remove <answer> and </answer> tags
    cleaned_text = re.sub(r"</?Answer>", "", answer_text)

    # Check if contains thinking block
    think_pattern = r"<Analyze>(.*?)</Analyze>\s*"
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)

    # If thinking block is found
    if think_match:
        # Get thinking block internal text
        think_content = think_match.group(1)
        # Extract content after thinking block
        after_think_content = cleaned_text[think_match.end() :].strip()

        # If there's content after thinking block, use the content after
        if after_think_content:
            cleaned_text = after_think_content
        else:
            # If no content after thinking block, look for answer within thinking block
            answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
            answer_match = re.search(answer_pattern, think_content, re.IGNORECASE)

            if answer_match:
                # Extract answer from thinking block
                extracted_answer = answer_match.group(1).strip()
                # Handle possible quotes and periods
                extracted_answer = re.sub(
                    r'^["\'](.*?)["\']\.?$', r"\1", extracted_answer
                )
                return extracted_answer

            # If no explicit "the answer is" in thinking block, try matching "Therefore, ..." or "Thus, ..."
            conclusion_pattern = r"(?:therefore|thus|so|hence|consequently|in conclusion)[,:]?\s+(.*?)(?:$|\.|\n)"
            conclusion_match = re.search(
                conclusion_pattern, think_content, re.IGNORECASE
            )

            if conclusion_match:
                return conclusion_match.group(1).strip()

    # Look for "the answer is" pattern in complete text
    answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
    answer_match = re.search(answer_pattern, cleaned_text, re.IGNORECASE)

    if answer_match:
        # Extract matched content
        extracted_answer = answer_match.group(1).strip()
        # Handle possible quotes
        extracted_answer = re.sub(r'^["\'](.*)["\']$', r"\1", extracted_answer)
        return extracted_answer

    # If no specific pattern found, return cleaned text
    return cleaned_text.strip()

def _normalize_answer(s):
    """Normalize answer text: convert to lowercase, remove punctuation, articles and extra spaces"""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _compute_exact(a_gold, a_pred):
    """Calculate exact match score"""
    return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))

def reward_hybridqa_exact_match(chat_history_str: str, ground_truth: Any) -> RewardInfo:
    pred = _clean_answer_tags(_extract_answer_from_response(chat_history_str))
    if not pred:
        return RewardInfo(reward=0.0, completed=0.0)

    is_match = _compute_exact(ground_truth, pred)
    return RewardInfo(reward=float(is_match), completed=1.0)