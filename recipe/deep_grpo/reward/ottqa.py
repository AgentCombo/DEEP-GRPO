import logging
import os
import re
import string

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_answer_from_response(model_answer):
    """Extract final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""

    # Case 1: <answer>...</answer> tag
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)
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

def _clean_answer_tags(answer_text):
    """Clean answer tags and extract the most relevant answer text"""
    if not answer_text:
        return ""

    # Remove <answer> and </answer> tags
    cleaned_text = re.sub(r"</?Answer>", "", answer_text)

    # Check if it contains a thinking block
    think_pattern = r"<Analyze>(.*?)</Analyze>\s*"
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)

    # If thinking block is found
    if think_match:
        # Get the text inside the thinking block
        think_content = think_match.group(1)
        # Extract content after the thinking block
        after_think_content = cleaned_text[think_match.end() :].strip()

        # If there's content after the thinking block, use that
        if after_think_content:
            cleaned_text = after_think_content
        else:
            # If there's no content after the thinking block, look for an answer inside it
            answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
            answer_match = re.search(answer_pattern, think_content, re.IGNORECASE)

            if answer_match:
                # Extract answer from inside the thinking block
                extracted_answer = answer_match.group(1).strip()
                # Handle possible quotes and periods
                extracted_answer = re.sub(
                    r'^["\'](.*?)["\']\.?$', r"\1", extracted_answer
                )
                return extracted_answer

            # If there's no explicit "the answer is" in the thinking block, try matching "Therefore, ..." or "Thus, ..."
            conclusion_pattern = r"(?:therefore|thus|so|hence|consequently|in conclusion)[,:]?\s+(.*?)(?:$|\.|\n)"
            conclusion_match = re.search(
                conclusion_pattern, think_content, re.IGNORECASE
            )

            if conclusion_match:
                return conclusion_match.group(1).strip()

    # Look for "the answer is" pattern in the full text
    answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
    answer_match = re.search(answer_pattern, cleaned_text, re.IGNORECASE)

    if answer_match:
        # Extract the matching content
        extracted_answer = answer_match.group(1).strip()
        # Handle possible quotes
        extracted_answer = re.sub(r'^["\'](.*)["\']$', r"\1", extracted_answer)
        return extracted_answer

    # If no specific pattern was found, return the cleaned text
    return cleaned_text.strip()

def _normalize_answer(s):
    """Normalize answer text: lowercase, remove punctuation, articles, and extra whitespace"""

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

def reward_ottqa_exact_match(chat_history_str: str, ground_truth: str) -> RewardInfo:
    pred_raw = _extract_answer_from_response(chat_history_str)
    pred = _clean_answer_tags(pred_raw)

    if not pred:
      return RewardInfo(reward=0.0, completed=0.0)

    em = _compute_exact(ground_truth, pred)
    return RewardInfo(reward=float(em), completed=1.0)
