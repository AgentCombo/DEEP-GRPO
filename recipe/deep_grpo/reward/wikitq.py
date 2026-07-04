import logging
import os
import re

from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_predicted_answer(model_answer):
    """Extract predicted answer from model's response"""
    if not model_answer:
        return None
    model_answer = model_answer.replace("**", "")
    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r"<Answer>(.*?)</Answer>"
    answer_tag_match = re.findall(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)

    if answer_tag_match:
        model_answer = answer_tag_match[-1].strip()

        return model_answer.split("Answer:")[-1].strip(".").strip()
        # if 'Answer:' in model_answer:
        #     match = re.findall(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
        #     if match:
        #         return match[-1].strip()
        # else:
        #     return model_answer.strip() if model_answer else None
    else:
        match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
        if match:
            return match.group(1).strip()

def normalize_answer(answer):
    """
    Normalize answer for robust comparison, handling:
    - Numbers with/without commas
    - Units (km/h, pages, etc.)
    - Lists with different separators
    - Accents and special characters
    - Different date/time formats
    """
    if not answer:
        return ""

    # Convert to lowercase and strip spaces
    answer = str(answer).strip().lower()

    # Handle numerical values with commas or units
    numeric_with_units_match = re.match(
        r"^([\d,]+)\s*(days|years|pages|km\/h|mph)$", answer
    )
    if numeric_with_units_match:
        # Extract the numeric part and remove commas
        numeric_value = numeric_with_units_match.group(1).replace(",", "")
        unit = numeric_with_units_match.group(2)
        # Return standardized format
        return f"{numeric_value} {unit}"

    # Handle simple numbers with commas
    numeric_match = re.match(r"^[\d,]+$", answer)
    if numeric_match:
        return answer.replace(",", "")  # Remove commas

    # Handle time periods
    time_period_mapping = {
        "1 week": "7 days",
        "2 weeks": "14 days",
        "1 year": "12 months",
        # Add more mappings as needed
    }
    if answer in time_period_mapping:
        return time_period_mapping[answer]

    # Try to convert to float for numerical comparison
    try:
        num = float(answer.replace(",", ""))
        if num.is_integer():
            return str(int(num))
        return str(num)
    except:
        # Not a number, continue with further normalization
        pass

    # Handle lists with different separators
    if "|" in answer or "," in answer:
        items = re.split(r"[|,]\s*", answer)
        # Sort the items to handle different orders
        return "|".join(sorted([item.strip() for item in items if item.strip()]))

    # Remove accents for better character matching
    import unicodedata

    answer = "".join(
        c for c in unicodedata.normalize("NFKD", answer) if not unicodedata.combining(c)
    )

    # Final cleanup - remove unnecessary characters
    answer = re.sub(r"[^\w\s]", "", answer)  # Remove punctuation
    answer = " ".join(answer.split())  # Normalize whitespace

    return answer

def exact_match_enhanced(prediction, reference):
    """Enhanced exact match function with various normalization techniques"""
    if prediction is None or reference is None:
        return 0

    # First try direct comparison after simple normalization
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if pred_norm == ref_norm:
        return 1

    # Special case for numbers with units
    # Check if prediction is just the number part of reference with units
    pred_num_match = re.match(r"^(\d+)$", pred_norm)
    ref_units_match = re.match(r"^(\d+)\s+([a-z/]+)$", ref_norm)

    if (
        pred_num_match
        and ref_units_match
        and pred_num_match.group(1) == ref_units_match.group(1)
    ):
        return 1

    # Check if both are lists but with different separators
    pred_items = set(re.split(r"[|,]\s*", prediction.lower()))
    ref_items = set(re.split(r"[|,]\s*", reference.lower()))

    if len(pred_items) > 1 and len(ref_items) > 1 and pred_items == ref_items:
        return 1

    return 0

def evaluate_answer(chat_history_str: str, ground_truth: str):
    predicted_answer = extract_predicted_answer(chat_history_str)

    if predicted_answer is None:
        return RewardInfo(reward=0.0, completed=0.0)

    is_match = exact_match_enhanced(predicted_answer, ground_truth)
    score = 1.0 if is_match else 0.0
    return RewardInfo(reward=score, completed=1)