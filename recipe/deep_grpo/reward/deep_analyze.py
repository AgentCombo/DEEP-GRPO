from typing import Optional

import re
import json
import logging
import os

import asyncio

from recipe.deep_grpo.protocol import RewardInfo
from recipe.deep_grpo.utils import call_teacher_with_retry
from recipe.deep_grpo.reward.prompts.deep_analyze import REPORT_SOUNDNESS_EVALUATE_PROMPT, REPORT_QUALITY_EVALUATE_PROMPT, ANSWER_CORRECTNESS_EVALUATE_PROMPT, RESPONSE_QUALITY_PROMPT


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _check_format(model_response: str) -> bool:
    if (
        "<Analyze>" not in model_response
        or "</Analyze>" not in model_response
        or "<Answer>" not in model_response
        or "</Answer>" not in model_response
    ):
        return False
    return True

def _check_code_format(text: str) -> bool:
    code_blocks = re.findall(r"<Code>(.*?)</Code>", text, re.DOTALL)
    if not code_blocks:
        return True
    for block in code_blocks:
        block = block.strip()
        if not (block.startswith("```python") and block.endswith("```")):
            return False
    return True

def _extract_answer(text):
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None

def _extract_model_response(text: str) -> Optional[str]:
    marker = "<Analyze>"
    start_index = text.find(marker)
    if start_index != -1:
        return text[start_index:]
    else:
        return None

def _compute_code_pass_ratio_reward(model_response: str, error_marker: str = "[Error]:") -> RewardInfo:
    observations = re.findall(r"<Execute>\s*(.*?)\s*</Execute>", model_response, flags=re.DOTALL)
    if not observations:
        return RewardInfo(
            reward=0.0,
            completed=1
        )
    total = len(observations)
    success = sum(1 for obs in observations if error_marker not in obs)
    return RewardInfo(
        reward=success / total,
        completed=1
    )

# def _parse_answer_correctness_reply(reply: str) -> Optional[float]:
#     matches = re.findall(
#         r"###\s*Final\s*Score\s*:\s*([01])\b",
#         reply,
#         re.IGNORECASE | re.MULTILINE,
#     )
#     if not matches:
#         raise ValueError("No '### Final Score:' line found in judgement reply")
#     val_str = matches[-1]
#     raw = float(val_str)
#     return raw


def _parse_answer_correctness_reply(reply: str) -> float:
    matches = re.findall(
        r"(?:###\s*)?Final\s*Score\s*:\s*([\d\.]+)", 
        reply,
        re.IGNORECASE | re.MULTILINE,
    )
    
    if not matches:
        raise ValueError("No '### Final Score:' line found in judgement reply")
    
    val_str = matches[-1]
    
    try:
        val = float(val_str)
    except ValueError:
        raise ValueError(f"Invalid score format (not a number): {val_str}")

    if val not in [0.0, 1.0]:
        raise ValueError(f"Invalid score value: {val}. Judge must output exactly 0 or 1.")

    return val

def _parse_response_quality_reply(reply: str) -> float:
    matches = re.findall(
        r"^SCORE\s*:\s*([0-4])\b",
        reply,
        re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        raise ValueError("No SCORE line found for response quality")
    val_str = matches[-1]
    raw = float(val_str)
    if not (0.0 <= raw <= 4.0):
        raise ValueError(f"Response quality SCORE out of range: {raw}")
    return raw / 4.0

# def _parse_report_soundness_reply(reply: str) -> dict:
#     matches = re.findall(r"```(?:json)?(.*?)```", reply, re.DOTALL | re.IGNORECASE)
#     if matches:
#         score_str = matches[-1].strip()
#     else:
#         score_str = reply.strip()
#     score_dict = json.loads(score_str)
#     required_keys = [
#         "faithfulness",
#         "instruction_answering",
#         "analytical_depth",
#         "process_coherence",
#         "visualization",
#     ]
#     for k in required_keys:
#         if k not in score_dict:
#             raise KeyError(f"Missing key in soundness json: {k}")
#         v = float(score_dict[k])
#         if not (0.0 <= v <= 4.0):
#             raise ValueError(f"Soundness score {k} out of range: {v}")
#     return score_dict

def _parse_report_soundness_reply(reply: str) -> dict:
    json_matches = re.findall(r"```json(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if json_matches:
        score_str = json_matches[-1].strip()
    else:
        all_matches = re.findall(r"```(.*?)```", reply, re.DOTALL)
        if all_matches:
            score_str = all_matches[-1].strip()
        else:
            score_str = reply.strip()
    score_dict = json.loads(score_str)
    required_keys = [
        "faithfulness",
        "instruction_answering",
        "analytical_depth",
        "process_coherence",
        "visualization",
    ]
    for k in required_keys:
        if k not in score_dict:
            raise KeyError(f"Missing key in soundness json: {k}")
        v = float(score_dict[k])
        if not (0.0 <= v <= 4.0):
            raise ValueError(f"Soundness score {k} out of range: {v}")
    return score_dict

# def _parse_report_quality_reply(reply: str) -> dict:
#     matches = re.findall(r"```(?:json)?(.*?)```", reply, re.DOTALL | re.IGNORECASE)
#     if matches:
#         score_str = matches[-1].strip()
#     else:
#         score_str = reply.strip()
#     score_dict = json.loads(score_str)
#     required_keys = [
#         "task_fulfillment",
#         "richness",
#         "soundness",
#         "visualization",
#         "readability",
#     ]
#     for k in required_keys:
#         if k not in score_dict:
#             raise KeyError(f"Missing key in report quality json: {k}")
#         v = float(score_dict[k])
#         if not (0.0 <= v <= 4.0):
#             raise ValueError(f"Report quality score {k} out of range: {v}")
#     return score_dict

def _parse_report_quality_reply(reply: str) -> dict:
    json_matches = re.findall(r"```json(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if json_matches:
        score_str = json_matches[-1].strip()
    else:
        all_matches = re.findall(r"```(.*?)```", reply, re.DOTALL)
        if all_matches:
            score_str = all_matches[-1].strip()
        else:
            score_str = reply.strip()
    score_dict = json.loads(score_str)
    required_keys = [
        "task_fulfillment",
        "richness",
        "soundness",
        "visualization",
        "readability",
    ]
    for k in required_keys:
        if k not in score_dict:
            raise KeyError(f"Missing key in report quality json: {k}")
        v = float(score_dict[k])
        if not (0.0 <= v <= 4.0):
            raise ValueError(f"Report quality score {k} out of range: {v}")
    return score_dict

async def _llm_as_judgement_answer_correctness(instruction: str, model_answer: str, ground_truth: str) -> RewardInfo:
    message = ANSWER_CORRECTNESS_EVALUATE_PROMPT.format(
        question=instruction, ground_truth=ground_truth, model_answer=model_answer
    )
    score, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_answer_correctness_reply,
        log_prefix="LLM Judge (answer_correctness)"
    )
    if score is None:
        logger.warning(f"Unrecognized answer correctness output after retries: {reply}")
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )
    return RewardInfo(
        reward=score, # 0.0 or 1.0
        completed=1,
        judgement_reply=reply
    )

async def _llm_as_judgement_response_quality(instruction: str, model_response: str, ground_truth: str) -> RewardInfo:
    message = RESPONSE_QUALITY_PROMPT.format(question=instruction, ground_truth=ground_truth, model_response=model_response)
    score, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_response_quality_reply,
        log_prefix="LLM Judge (response_quality)",
    )
    if score is None:
        logger.warning(f"Unrecognized response quality output after retries: {reply}")
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )
    return RewardInfo(
        reward=score,  # 0.0 ~ 1.0
        completed=1,
        judgement_reply=reply,
    )

async def _llm_as_judgement_report_soundness(
    instruction: str,
    model_response: str
) -> RewardInfo:
    message = REPORT_SOUNDNESS_EVALUATE_PROMPT.format(
        instruction=instruction,
        model_response=model_response,
    )
    score_dict, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_report_soundness_reply,
        log_prefix="LLM Judge (report_soundness)",
    )
    if score_dict is None:
        logger.warning(f"Unrecognized report quality output after retries: {reply}")
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )
    soundness_metric_weights = {
        "faithfulness": 2.0,
        "instruction_answering": 2.0,
        "process_coherence": 2.0,
        "analytical_depth": 2.0,
        "visualization": 1.0,
    }
    reward_dict = {
        metric: float(score_dict[metric]) / 4.0
        for metric in soundness_metric_weights
    }
    weighted_sum = sum(
        reward_dict[metric] * soundness_metric_weights[metric]
        for metric in soundness_metric_weights.keys()
    )
    average_score = weighted_sum / sum(soundness_metric_weights.values())
    return RewardInfo(
        reward=average_score, # 0.0 -> 1.0
        completed=1,
        judgement_reply=reply,
    )

async def _llm_as_judgement_report_quality(
    instruction: str,
    report: str
) -> RewardInfo:
    message = REPORT_QUALITY_EVALUATE_PROMPT.format(
        instruction=instruction,
        report=report,
    )
    score_dict, reply = await call_teacher_with_retry(
        message=message,
        parse_fn=_parse_report_quality_reply,
        log_prefix="LLM Judge (report_quality)",
    )
    if score_dict is None:
        logger.warning(f"Unrecognized report quality output after retries: {reply}")
        return RewardInfo(
            reward=0.0,
            completed=0,
            judgement_reply=reply,
        )
    report_quality_metric_weights = {
        "task_fulfillment": 1.0,
        "richness": 1.0,
        "soundness": 1.0,
        "visualization": 1.0,
        "readability": 1.0,
    }
    reward_dict = {
        metric: float(score_dict[metric]) / 4.0
        for metric in report_quality_metric_weights
    }
    weighted_sum = sum(
        reward_dict[metric] * report_quality_metric_weights[metric]
        for metric in report_quality_metric_weights.keys()
    )
    average_score = weighted_sum / sum(report_quality_metric_weights.values())
    return RewardInfo(
        reward=average_score, # 0.0 -> 1.0
        completed=1,
        judgement_reply=reply,
    )

def _extract_tableqa_answer(model_response: str):
    last_pos = model_response.rfind("<Answer>")
    if last_pos == -1:
        return None
    remaining_text = model_response[last_pos:]
    pattern = r"<Answer>(.*?)</Answer>"
    match = re.search(pattern, remaining_text, re.DOTALL)
    result = None
    if match:
        result = match.group(1).strip()
    if result is not None and "Answer:" in result:
        result = result.split("Answer:")[1].strip()
    return result

def _compute_tableqa_accuracy_reward(model_response: str, ground_truth: str) -> RewardInfo:
    answer = _extract_tableqa_answer(model_response)

    if answer is None:
        return RewardInfo(reward=0.0,
                            completed=1)

    if answer.strip().lower() == ground_truth.strip().lower():
        return RewardInfo(reward=1.0,
                            completed=1)
    else:
        return RewardInfo(reward=0.0,
                            completed=1)

async def compute_tableqa_reward(instruction: str, chat_history_str: str, ground_truth: str) -> RewardInfo:
    # format check
    # if not _check_format(chat_history_str):
    #     return RewardInfo(
    #         reward=-1.0,
    #         completed=1
    #     )

    model_response = _extract_model_response(chat_history_str)
    if model_response is None:
        return RewardInfo(
            reward=0.0,
            completed=1.0
        )
    
    # correctness reward
    reward_answer_correctness = _compute_tableqa_accuracy_reward(model_response, ground_truth)

    return reward_answer_correctness

    # response quality reward
    reward_response_quality = await _llm_as_judgement_response_quality(
                    instruction=instruction,
                    model_response=model_response,
                    ground_truth=ground_truth
                )
    
    # fuse rewards
    rewards_weights = {
        "answer_correctness": 1.0,
        "response_quality": 1.0,
    }
    
    final_reward = (
        reward_answer_correctness.reward * rewards_weights["answer_correctness"] +
        reward_response_quality.reward * rewards_weights["response_quality"]
    ) / sum(rewards_weights.values()) # 0.0 -> 1.0
    completed = 1 if (reward_answer_correctness.completed == 1 and reward_response_quality.completed == 1) else 0
    return RewardInfo(
        reward=final_reward,
        completed=completed,
        judgement_reply=f"[Response Quality Judgement]\n{reward_response_quality.judgement_reply or ''}"
    )

async def compute_datatask_reward(instruction: str, chat_history_str: str, ground_truth: str) -> RewardInfo:
    # format check
    # if not _check_format(chat_history_str):
    #     return RewardInfo(
    #         reward=-1.0,
    #         completed=1
    #     )
    
    model_response = _extract_model_response(chat_history_str)
    if model_response is None:
        return RewardInfo(
            reward=0.0,
            completed=1
        )

    # assert model_response is not None # pass format check, the model response should be extracted

    # code format check
    # if not _check_code_format(model_response):
    #     return RewardInfo(
    #         reward=-1.0,
    #         completed=1
    #     )
    
    # code pass ratio reward
    reward_code_pass_ratio = _compute_code_pass_ratio_reward(model_response)

    # answer correctness reward and response quality reward
    model_answer = _extract_answer(model_response)
    if model_answer is None:
        return RewardInfo(
            reward=0.0,
            completed=1
        )
    
    reward_answer_correctness = await _llm_as_judgement_answer_correctness(
                    instruction=instruction,
                    model_answer=model_answer,
                    ground_truth=ground_truth
                )
    
    if reward_answer_correctness.reward == 0.0:
        return RewardInfo(
            reward=0.0,
            completed=reward_answer_correctness.completed,
            judgement_reply=reward_answer_correctness.judgement_reply
        )
    
    final_reward = 0.5 * reward_answer_correctness.reward + 0.5 * reward_code_pass_ratio.reward
    completed = 1 if (reward_code_pass_ratio.completed == 1 and reward_answer_correctness.completed == 1) else 0

    # final_reward = reward_answer_correctness.reward
    # completed = reward_answer_correctness.completed

    judgement_reply = reward_answer_correctness.judgement_reply

    return RewardInfo(
        reward=final_reward,
        completed=completed,
        judgement_reply=judgement_reply
    )

    rewards_weights = {
        "code_pass_ratio": 0.3,
        "answer_correctness": 0.7,
    }

    final_reward = (
        reward_code_pass_ratio.reward * rewards_weights["code_pass_ratio"] +
        reward_answer_correctness.reward * rewards_weights["answer_correctness"]
    ) / sum(rewards_weights.values())

    completed = 1 if (reward_code_pass_ratio.completed == 1 and reward_answer_correctness.completed == 1) else 0

    judgement_reply = f"[Answer Correctness Judgement]\n{reward_answer_correctness.judgement_reply or ''}"

    return RewardInfo(
        reward=final_reward,
        completed=completed,
        judgement_reply=judgement_reply
    )

    reward_answer_correctness, reward_response_quality = await asyncio.gather(
                _llm_as_judgement_answer_correctness(
                    instruction=instruction,
                    model_answer=model_answer,
                    ground_truth=ground_truth
                ),
                _llm_as_judgement_response_quality(
                    instruction=instruction,
                    model_response=model_response,
                    ground_truth=ground_truth
                )
            )

    # fuse rewards
    rewards_weights = {
        "code_pass_ratio": 1.0,
        "answer_correctness": 1.0,
        "response_quality": 1.0,
    }
    
    final_reward = (
        reward_code_pass_ratio.reward * rewards_weights["code_pass_ratio"] +
        reward_answer_correctness.reward * rewards_weights["answer_correctness"] +
        reward_response_quality.reward * rewards_weights["response_quality"]
    ) / sum(rewards_weights.values()) # 0.0 -> 1.0
    completed = 1 if (reward_code_pass_ratio.completed == 1 and reward_answer_correctness.completed == 1 and reward_response_quality.completed == 1) else 0
    # completed = 1 if (reward_answer_correctness.completed == 1 and reward_response_quality.completed == 1) else 0
    judgement_reply = (
        f"[Answer Correctness Judgement]\n{reward_answer_correctness.judgement_reply or ''}\n\n"
        f"[Response Quality Judgement]\n{reward_response_quality.judgement_reply or ''}"
    )
    return RewardInfo(
        reward=final_reward,
        completed=completed,
        judgement_reply=judgement_reply
    )

async def compute_research_task_reward(instruction: str, chat_history_str: str) -> RewardInfo:
    # format check
    # if not _check_format(chat_history_str):
    #     return RewardInfo(
    #         reward=-1.0,
    #         completed=1
    #     )

    model_response = _extract_model_response(chat_history_str)
    if model_response is None:
        return RewardInfo(
            reward=0.0,
            completed=1
        )

    # code format check
    # if not _check_code_format(model_response):
    #     return RewardInfo(
    #         reward=-1.0,
    #         completed=1
    #     )

    # code pass ratio reward
    reward_code_pass_ratio = _compute_code_pass_ratio_reward(model_response)

    # report quality reward and soundness reward
    report = _extract_answer(model_response)
    if report is None:
        return RewardInfo(
            reward=0.0,
            completed=1
        )

    reward_report_quality, reward_report_soundness = await asyncio.gather(
        _llm_as_judgement_report_quality(
            instruction=instruction,
            report=report
        ),
        _llm_as_judgement_report_soundness(
            instruction=instruction,
            model_response=model_response
        )
    )

    # fuse rewards
    rewards_weights = {
        "code_pass_ratio": 1.0,
        "report_quality": 1.0,
        "report_soundness": 1.0,
    }

    final_reward = (
        reward_code_pass_ratio.reward * rewards_weights["code_pass_ratio"] +
        reward_report_quality.reward * rewards_weights["report_quality"] +
        reward_report_soundness.reward * rewards_weights["report_soundness"]
    ) / sum(rewards_weights.values()) # 0.0 -> 1.0
    completed = 1 if (reward_code_pass_ratio.completed == 1 and reward_report_quality.completed == 1 and reward_report_soundness.completed == 1) else 0
    # completed = 1 if (reward_report_quality.completed == 1 and reward_report_soundness.completed == 1) else 0
    judgement_reply = (
        f"[Report Quality Judgement]\n{reward_report_quality.judgement_reply or ''}\n\n"
        f"[Report Soundness Judgement]\n{reward_report_soundness.judgement_reply or ''}"
    )

    return RewardInfo(
        reward=final_reward,
        completed=completed,
        judgement_reply=judgement_reply
    )


# async def compute_research_task_reward(instruction: str, chat_history_str: str) -> RewardInfo:
#     # format check
#     # if not _check_format(chat_history_str):
#     #     return RewardInfo(
#     #         reward=-1.0,
#     #         completed=1
#     #     )

#     model_response = _extract_model_response(chat_history_str)
#     if model_response is None:
#         return RewardInfo(
#             reward=0.0,
#             completed=1
#         )

#     # code format check
#     # if not _check_code_format(model_response):
#     #     return RewardInfo(
#     #         reward=-1.0,
#     #         completed=1
#     #     )

#     # code pass ratio reward
#     reward_code_pass_ratio = _compute_code_pass_ratio_reward(model_response)

#     # report quality reward and soundness reward
#     report = _extract_answer(model_response)
#     if report is None:
#         return RewardInfo(
#             reward=0.0,
#             completed=1
#         )
    
#     reward_report_quality = await _llm_as_judgement_report_quality(
#                     instruction=instruction,
#                     report=report
#                 )

#     return reward_report_quality