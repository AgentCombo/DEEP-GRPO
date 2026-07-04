import re
import json
import logging
import os

from recipe.deep_grpo.protocol import RewardInfo
from recipe.deep_grpo.utils import call_teacher_with_retry
from recipe.deep_grpo.reward.prompts.dsbench import ACCURACY_EVALUATE_PROMPT


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_answer(text):
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None

def _parse_accuracy_reply(reply: str) -> float:
    matches = re.findall(r"```(?:json)?(.*?)```", reply, re.DOTALL)
    if matches:
        json_str = matches[-1].strip()
    else:
        json_str = reply.strip()

    data = json.loads(json_str)
    if "correct" not in data:
        raise KeyError("Missing 'correct' in accuracy json")
    val = data["correct"]
    if not isinstance(val, bool):
        raise ValueError(f"'correct' must be a boolean, got {type(val)}: {val!r}")
    return 1.0 if val else 0.0

async def llm_as_judgement_accuracy(question: str,
                                    chat_history_str: str,
                                    ground_truth_answer: str
                                    ) -> RewardInfo:
    pred = extract_answer(chat_history_str) # can extract answer from the whole chat history
    if pred is None:
        return RewardInfo(reward=0,
                            completed=0)


    message = ACCURACY_EVALUATE_PROMPT.format(
        question=question,
        ground_truth=ground_truth_answer,
        predicted_answer=pred,
    )


    score, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_accuracy_reply,
        log_prefix="DSBench LLM Judge (accuracy)",
    )

    if score is None:
        logger.warning(
            "Unrecognized accuracy judge output after retries. "
            f"Raw reply: {reply}"
        )
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )

    return RewardInfo(
        reward=score, # 0.0 or 1.0
        completed=1,
        judgement_reply=reply,
    )