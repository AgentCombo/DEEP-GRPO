import re

import logging
import os

from recipe.deep_grpo.protocol import RewardInfo


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


def is_equal(response, label):
    if response is None:
        return False
    response = response.strip().strip('"').strip("'")
    label = label.strip().strip('"').strip("'")
    if response == label:
        return True
    else:
        try:
            return abs(float(response) - float(label)) < 1e-6
        except:
            return False

def extract_answer(text):
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None


def compute_score(chat_history_str: str, ground_truth) -> RewardInfo:
    answer_str = extract_answer(chat_history_str)
    if answer_str is None:
        return RewardInfo(reward=0,
                          completed=0)

    label_answers = {ans[0]: ans[1] for ans in ground_truth}
    answer_names, answers = extract_format(answer_str)
    extracted_answers = dict(zip(answer_names, answers))
    completed = 1
    for anser_name in label_answers:
        if extracted_answers.get(anser_name) is None:
            completed = 0
            break
    correct_answers = {anser_name: is_equal(extracted_answers.get(anser_name), label_answers[anser_name]) for anser_name in label_answers.keys()}
    correctness_reward = 1 if all(correct_answers.values()) else 0
    return RewardInfo(reward=correctness_reward,
                      completed=completed)