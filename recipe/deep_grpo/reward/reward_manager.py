from typing import List

import asyncio

from recipe.deep_grpo.reward import dabench
from recipe.deep_grpo.reward import deep_analyze
from recipe.deep_grpo.reward import dsbench_data_analysis
from recipe.deep_grpo.reward import dsbench_data_modeling
from recipe.deep_grpo.reward import gsm8k
from recipe.deep_grpo.reward import math
from recipe.deep_grpo.reward import math_original
from recipe.deep_grpo.reward import aime24
from recipe.deep_grpo.reward import amc
from recipe.deep_grpo.reward import minerva
from recipe.deep_grpo.reward import olympiadbench
from recipe.deep_grpo.reward import deepscaler
from recipe.deep_grpo.reward import math_verify
from recipe.deep_grpo.reward import ds1000
from recipe.deep_grpo.reward import wikitq
from recipe.deep_grpo.reward import hybridqa
from recipe.deep_grpo.reward import multihiertt
from recipe.deep_grpo.reward import ottqa
from recipe.deep_grpo.reward import finqa
from recipe.deep_grpo.reward import dabstep_research
from recipe.deep_grpo.reward import dabstep
from recipe.deep_grpo.reward import qa_em_format
from recipe.deep_grpo.protocol import RewardInfo, FINISH_REASON, Node


async def score_node(node: Node, tokenizer, is_validation: bool = False) -> RewardInfo:
    if node.finish_reason != FINISH_REASON.COMPLETED: # first check whether the response is complete, if not, assign 0 reward
        return RewardInfo(reward=0.0, completed=0, finished=0)

    loop = asyncio.get_running_loop()

    chat_history_str = await loop.run_in_executor(
            None, lambda: tokenizer.decode(node.prompt_ids + node.response_ids, skip_special_tokens=False)
    ) # include instruction, model response and special tokens

    data_instance = node.data_instance

    data_source = data_instance["data_source"]

    if data_source == "datatask":
        result = await deep_analyze.compute_datatask_reward(instruction=data_instance["extra_info"]["instruction"],
                                                            chat_history_str=chat_history_str,
                                                            ground_truth=data_instance["reward_model"]["ground_truth"]
                                                            )
    
    elif data_source == "reasoning-table":
        result = await deep_analyze.compute_tableqa_reward(instruction=data_instance["extra_info"]["instruction"],
                                                           chat_history_str=chat_history_str,
                                                           ground_truth=data_instance["reward_model"]["ground_truth"]
                                                          )
        
    elif data_source == "research":
        result = await deep_analyze.compute_research_task_reward(instruction=data_instance["extra_info"]["instruction"],
                                                                 chat_history_str=chat_history_str,
                                                                )
        
    elif data_source == "DSBench_data_analysis":
        result = await dsbench_data_analysis.llm_as_judgement_accuracy(question=data_instance["extra_info"]["question"],
                                                                        chat_history_str=chat_history_str,
                                                                        ground_truth_answer=data_instance["reward_model"]["ground_truth"]
                                                                    )
    elif data_source == "DSBench_data_modeling":
        result = await dsbench_data_modeling.evaluate_submission(chat_history_str=chat_history_str,
                                                                eval_script_path=data_instance["extra_info"]["evaluation_script"],
                                                                submission_file=data_instance["reward_model"]["expected_file"],
                                                                ground_truth_file=data_instance["reward_model"]["ground_truth"],
                                                                baseline_result=data_instance["reward_model"]["baseline_result"],
                                                                gold_result=data_instance["reward_model"]["gold_result"])
    elif data_source == "DABench":
        result = dabench.compute_score(chat_history_str=chat_history_str,
                                       ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "GSM8K":
        result = gsm8k.compute_score(chat_history_str=chat_history_str,
                                     ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "DigitalLearningGmbH/MATH-lighteval":
        result = math.compute_score(chat_history_str=chat_history_str,
                                    ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "MATH":
        result = math.compute_score(chat_history_str=chat_history_str,
                                    ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "AIME24":
        result = aime24.compute_score(chat_history_str=chat_history_str,
                                      ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "AMC":
        result = amc.compute_score(chat_history_str=chat_history_str,
                                   ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "Minerva":
        result = minerva.compute_score(chat_history_str=chat_history_str,
                                       ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "OlympiadBench":
        result = olympiadbench.compute_score(chat_history_str=chat_history_str,
                                             ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "DeepScaleR":
        result = await math_verify.compute_score(chat_history_str=chat_history_str,
                                          ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "ds1000":
        result = await ds1000.evaluate_ds1000(chat_history_str=chat_history_str, code_context=data_instance["extra_info"]["code_context"])

    elif data_source == "wikitq":
        result = wikitq.evaluate_answer(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "hybridqa":
        result = hybridqa.reward_hybridqa_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "multihiertt":
        result = multihiertt.reward_multihiertt_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "ottqa":
        result = ottqa.reward_ottqa_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "finqa":
        result = finqa.reward_finqa(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "hotpotqa" or data_source == "2wikimultihopqa" or data_source == "musique" or data_source == "bamboogle":
        if is_validation:
            result = qa_em_format.compute_score_em(solution_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"],
                                                   structure_format_score=0, final_format_score=0, retrieval_score=0)
        else:
            result = qa_em_format.compute_score_em(solution_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "dabstep_research":
        result = await dabstep_research.llm_as_judgement_dabstep_report_quality(instruction=data_instance["extra_info"]["instruction"],
                                                                   chat_history_str=chat_history_str,
                                                                   checklist=data_instance["reward_model"]["ground_truth"]
                                                                )
    
    elif data_source == "dabstep":
        result = await dabstep.llm_as_judgement_dabstep_dummy(instruction=data_instance["extra_info"]["instruction"],
                                                              chat_history_str=chat_history_str,
                                                              ground_truth=data_instance["reward_model"]["ground_truth"]
                                                            )

    else:
        raise ValueError(f"Unknown data source {data_source}")
    
    result.finished = 1 # we have checked whether the response is complete at the beginning
    
    return result