import os
import logging
from recipe.deep_grpo.protocol import RewardInfo

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def llm_as_judgement_dabstep_dummy(
    instruction: str,
    chat_history_str: str,
    ground_truth: str,
) -> RewardInfo:
    return RewardInfo(
        reward=-1.0,
        completed=0,
        judgement_reply="DABstep dummy reward: leaderboard-only scoring (reward=-1).",
    )
