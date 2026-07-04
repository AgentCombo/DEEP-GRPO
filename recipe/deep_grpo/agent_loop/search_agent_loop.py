import logging
import os
from typing import Any, Dict
from uuid import uuid4

import regex as re

from recipe.deep_grpo.protocol import FINISH_REASON, Node
from recipe.deep_grpo.agent_loop.deep_grpo_agent_loop import DeepGRPOAgentLoop
from recipe.deep_grpo.agent_loop.retriever import Retriever


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchAgentLoop(DeepGRPOAgentLoop):
    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        
        super().init_class(config, tokenizer)
        
        cls.stop_words = ["</search>"]

        cls.add_generation_prompt = True

        cls.retriever = Retriever(search_url=os.getenv("retriever_url", "http://127.0.0.1:8000/retrieve"), max_tool_response_length=config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length)
    
    def _extract_retrieve_query(self, response: str):
        match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        prompt_ids = parent_node.prompt_ids + parent_node.response_ids
        response_mask = []
        finish_reason = None

        sampling_params_with_stop = sampling_params.copy()
        sampling_params_with_stop['stop'] = self.stop_words

        assert len(prompt_ids) < self.max_model_len
        result = await self.server_manager.generate_with_finish_reason(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params_with_stop
        )
        response_ids = result["token_ids"]
        response_mask += [1] * len(response_ids)

        if result["finish_reason"] == "length":
            finish_reason = FINISH_REASON.EXCEED_LENGTH
        elif result["finish_reason"] == "stop":
            stop_reason = result.get("stop_reason")

            if stop_reason in self.stop_words:
                if len(prompt_ids) + len(response_ids) == self.max_model_len:
                    finish_reason = FINISH_REASON.EXCEED_LENGTH
                else:
                    response = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False))
                    query = self._extract_retrieve_query(response)
                    if query:
                        retrieve_result = await self.retriever.search(query)
                        retrieve_block = f"\n\n<information>{retrieve_result}</information>\n\n"
                        retrieve_ids = await self.loop.run_in_executor(None, lambda: self.tokenizer.encode(retrieve_block, add_special_tokens=False))
                        if len(prompt_ids) + len(response_ids) + len(retrieve_ids) > self.max_model_len:
                            finish_reason = FINISH_REASON.EXCEED_LENGTH
                        else:
                            response_ids += retrieve_ids
                            response_mask += [0] * len(retrieve_ids)
                            if len(prompt_ids) + len(response_ids) == self.max_model_len:
                                finish_reason = FINISH_REASON.EXCEED_LENGTH
                            else:
                                finish_reason = FINISH_REASON.STOP
                    else:
                        finish_reason = FINISH_REASON.COMPLETED # Not valid search action, stop the trajectory

            else: # Generate <eos>
                if len(prompt_ids) + len(response_ids) == self.max_model_len:
                    finish_reason = FINISH_REASON.EXCEED_LENGTH
                else:
                    finish_reason = FINISH_REASON.COMPLETED
        else:
            raise RuntimeError(f"Unexpected finish_reason: {result['finish_reason']}")

        total_len = len(prompt_ids) + len(response_ids)
        assert total_len <= self.max_model_len, f"node length {total_len} > max_model_len {self.max_model_len}"
        
        if finish_reason == FINISH_REASON.STOP or finish_reason == FINISH_REASON.COMPLETED:
            assert len(prompt_ids) + len(response_ids) < self.max_model_len, f"STOP/COMPLETED node length {len(prompt_ids) + len(response_ids)} >= max_model_len {self.max_model_len}"

        return Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            data_instance=parent_node.data_instance,
            finish_reason=finish_reason,
            num_turns=parent_node.num_turns + 1
        )