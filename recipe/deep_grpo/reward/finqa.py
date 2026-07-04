import logging
import os
import re
from typing import Tuple, Optional

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _normalize_answer(answer: str) -> Tuple[Optional[float], str, bool]:
    """
    Normalize answer format, identify value type and standardize format

    Returns:
        (numeric_value, unit/type, is_percentage)
    """
    if answer is None:
        return None, "", False

    original = answer
    answer = answer.strip().lower()

    # Detect units and format
    is_percentage = "%" in answer
    has_dollar = "$" in answer
    has_million = "million" in answer.lower()
    has_billion = "billion" in answer.lower()

    # Extract numeric part
    num_pattern = r"[-+]?[0-9]*\.?[0-9]+"
    num_match = re.search(num_pattern, answer)

    if not num_match:
        return None, original, False

    try:
        # Extract and convert number
        num_value = float(num_match.group(0))

        # Prepare return value
        if is_percentage:
            return num_value, "%", True
        elif has_million:
            return num_value, "million", False
        elif has_billion:
            return num_value, "billion", False
        elif has_dollar:
            return num_value, "$", False
        else:
            return num_value, "", False
    except:
        return None, original, False

def _extract_answer_from_response(model_answer: str) -> str:
    """Extract answer from model response"""
    if not model_answer:
        return ""

    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r"<Answer>(.*?)</Answer>"

    tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)
    if tag_match:
        model_answer = tag_match.group(1).strip()

    # Try to match content after "Answer:"
    answer_pattern = r"Answer:\s*(.*?)(?:(?:\n\s*\n)|(?:\n\s*Used cells:)|$)"
    match = re.findall(answer_pattern, model_answer, re.DOTALL)
    if match:
        # Extract match content and clean it
        answer_text = match[-1].strip()
        answer_text = (
            answer_text.replace("</answer>", "")
            .replace("<answer>", "")
            .replace("Answer:", "")
        )
        return answer_text

    # Fallback approaches: try various answer formats

    # 1. Look for "Answer:" or similar patterns
    for pattern in [r"Answer:\s*(.*?)(?:\n|$)", r"answer:\s*(.*?)(?:\n|$)"]:
        # match = re.search(pattern, model_answer)
        # if match:
        #     return match.group(1).strip()

        answer_matches = re.findall(pattern, model_answer, re.DOTALL)
        if answer_matches:
            return answer_matches[-1].strip()

    # 2. Look for formatted numbers (possibly with currency symbols, percentages or units)
    number_patterns = [
        r"([$¥€£]?\s*[-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
        r"([-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
        r"([-+]?[0-9,]*\.?[0-9]+\%)",
    ]

    for pattern in number_patterns:
        match = re.findall(pattern, model_answer)
        if match:
            return match[-1].strip()

    # 3. Extract the last line that might contain an answer
    lines = model_answer.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.lower().startswith(
            ("thus", "therefore", "so", "hence")
        ):
            # Try to extract numeric expression from this line
            number_match = re.findall(
                r"([$¥€£]?\s*[-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
                line,
            )
            if number_match:
                return number_match[-1].strip()
            break

    return ""

def _check_answer_correctness(
    model_answer: str, expected_answer: str
) -> Tuple[bool, str]:
    """Check if model answer is correct, supporting various numeric format comparisons"""
    # Extract model's answer
    extracted_answer = _extract_answer_from_response(model_answer)
    if not extracted_answer:
        return False, ""

    # Normalize and extract numeric values
    model_value, model_unit, model_is_percent = _normalize_answer(extracted_answer)
    expected_value, expected_unit, expected_is_percent = _normalize_answer(
        expected_answer
    )

    if model_value is None or expected_value is None:
        # If numbers cannot be extracted, fall back to exact string matching
        return (
            extracted_answer.strip().lower() == expected_answer.strip().lower(),
            extracted_answer,
        )

    # Check if values and units are equivalent (considering unit variations)
    # For identical values, don't require exactly matching units
    if abs(model_value - expected_value) < 0.01:
        # For very close values, consider correct
        return True, extracted_answer

    # Unit conversion and comparison
    is_same_unit_type = False

    # Check unit type consistency
    if model_is_percent == expected_is_percent:
        is_same_unit_type = True
    # Special handling for "$X" and "X million" cases
    elif model_unit == "$" and expected_unit == "million":
        model_value = model_value * 1000000
        is_same_unit_type = True
    elif expected_unit == "$" and model_unit == "million":
        expected_value = expected_value * 1000000
        is_same_unit_type = True
    # Special handling for "$X" and "X billion" cases
    elif model_unit == "$" and expected_unit == "billion":
        model_value = model_value * 1000000000
        is_same_unit_type = True
    elif expected_unit == "$" and model_unit == "billion":
        expected_value = expected_value * 1000000000
        is_same_unit_type = True
    # Special handling for "$X million" and "X" cases (assuming X alone means millions)
    elif model_unit == "$" and model_unit == "million" and expected_unit == "":
        is_same_unit_type = True
    elif expected_unit == "$" and expected_unit == "million" and model_unit == "":
        is_same_unit_type = True

    if not is_same_unit_type:
        # Unit types inconsistent, but if values are identical, still consider a match
        if abs(model_value - expected_value) < 0.01:
            return True, extracted_answer
        return False, extracted_answer

    # Allow percentage precision errors (round to 1 decimal place)
    if model_is_percent and expected_is_percent:
        # Round to 1 decimal place before comparing
        model_rounded = round(model_value, 1)
        expected_rounded = round(expected_value, 1)
        return abs(model_rounded - expected_rounded) < 0.15, extracted_answer

    # Handle monetary amounts
    if (
        model_unit == "$"
        or model_unit == "million"
        or model_unit == "billion"
        or expected_unit == "$"
        or expected_unit == "million"
        or expected_unit == "billion"
    ):
        # Compare after converting to the same unit
        if abs(model_value - expected_value) / max(abs(expected_value), 1) < 0.05:
            return True, extracted_answer
    else:
        # General number comparison, allow small error margin
        relative_error = abs(model_value - expected_value) / max(abs(expected_value), 1)
        if relative_error < 0.05:  # 5% relative error margin
            return True, extracted_answer

    return False, extracted_answer

def reward_finqa(chat_history_str: str, ground_truth: str) -> RewardInfo:
    ok, extracted = _check_answer_correctness(chat_history_str, ground_truth)

    if not extracted:
        return RewardInfo(reward=0.0, completed=0.0)

    return RewardInfo(reward=1.0 if ok else 0.0, completed=1.0)
