import os
import logging
from recipe.deep_grpo.protocol import RewardInfo
from recipe.deep_grpo.reward.math_grader import boxed_reward_fn


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_score(chat_history_str, ground_truth) -> RewardInfo:
    info, reward = boxed_reward_fn(chat_history_str, ground_truth, fast=False)
    completed = int(info["formatted"])
    return RewardInfo(reward=reward, completed=completed)