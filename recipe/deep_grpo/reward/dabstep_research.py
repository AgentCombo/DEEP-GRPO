import re
import json
import logging
import os
from typing import Optional, Dict, Any

from recipe.deep_grpo.protocol import RewardInfo
from recipe.deep_grpo.reward.prompts.dabstep_research import PROMPT
from recipe.deep_grpo.utils import call_teacher_with_retry


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_report(text: str) -> Optional[str]:
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _parse_quality_reply(reply: str) -> float:
    match = re.search(r"```(?:json)?(.*?)```", reply, re.DOTALL)
    score_str = match.group(1).strip() if match else reply
    score_dict: Dict[str, Any] = json.loads(score_str)

    avg_score = (float(score_dict["Content"]) + float(score_dict["Format"])) / 2
    return float(avg_score)


async def llm_as_judgement_dabstep_report_quality(
    instruction: str,
    chat_history_str: str,
    checklist: str,
) -> RewardInfo:
    report = extract_report(chat_history_str)

    if report is None:
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply="",
        )

    message  = PROMPT.format(
        instruction=instruction,
        checklist=checklist,
        report=report,
    )

    score, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_quality_reply,
        temperature_schedule=(0.0,),
        max_attempts_per_temperature=1,
        log_prefix="DABStep Research LLM Judge (report quality)",
    )

    if score is None:
        logger.warning(
            "Unrecognized judge output. "
            f"Reply:\n{reply}"
        )
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )

    return RewardInfo(
        reward=float(score),
        completed=1,
        judgement_reply=reply,
    )