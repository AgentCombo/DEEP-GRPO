import logging
import os
from typing import Any, Dict, List
from uuid import uuid4

import regex as re

from recipe.deep_grpo.agent_loop.code_executor import CodeExecutor
from recipe.deep_grpo.protocol import FINISH_REASON, Node
from recipe.deep_grpo.agent_loop.outputs import TSValAgentLoopOutput
from recipe.deep_grpo.agent_loop.deep_grpo_agent_loop import DeepGRPOAgentLoop

from verl.utils.rollout_trace import rollout_trace_op


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepAnalyzeAgentLoop(DeepGRPOAgentLoop):
    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        
        super().init_class(config, tokenizer)
        
        cls.code_executor = CodeExecutor(max_tool_response_length=config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length)
        cls.stop_word_code = "</Code>"
        cls.stop_words = [cls.stop_word_code]
        cls.stop_token_ids = [tokenizer.convert_tokens_to_ids(word) for word in cls.stop_words]
        cls.stop_token_map = {
            tokenizer.convert_tokens_to_ids(cls.stop_word_code): "code",
        }

        cls.add_generation_prompt = False # avoid double <Assistant>

    def _extract_code(self, response: str):
        code_match = re.search(r"<Code>(.*?)</Code>", response, re.DOTALL)
        if not code_match:
            return None
        code_content = code_match.group(1).strip()
        md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
        code_str = md_match.group(1).strip() if md_match else code_content
        return code_str

    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        prompt_ids = parent_node.prompt_ids + parent_node.response_ids
        workspace = parent_node.data_instance["extra_info"]["workspace"]
        response_mask = []
        finish_reason = None

        sampling_params_with_stop = sampling_params.copy()
        sampling_params_with_stop['stop_token_ids'] = self.stop_token_ids

        assert len(prompt_ids) < self.max_model_len
        result = await self.server_manager.generate_with_finish_reason(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params_with_stop
        )
        response_ids = result["token_ids"]
        response_mask += [1] * len(response_ids)

        if result["finish_reason"] == "length":
            finish_reason = FINISH_REASON.EXCEED_LENGTH
        elif result["finish_reason"] == "stop":
            stop_reason_id = result.get("stop_reason")

            if stop_reason_id in self.stop_token_ids: # Generate <Code></Code>
                stop_type = self.stop_token_map[stop_reason_id]

                if len(prompt_ids) + len(response_ids) == self.max_model_len:
                    finish_reason = FINISH_REASON.EXCEED_LENGTH
                else:
                    response = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False))
                    if stop_type == "code":
                        code = self._extract_code(response)
                        if code is None:
                            finish_reason = FINISH_REASON.COMPLETED
                        else:
                            execution_result = await self.code_executor.execute_code(code_str=code, workspace=workspace)
                            execution_block = f"\n<Execute>\n{execution_result}\n</Execute>\n"
                            execution_ids = await self.loop.run_in_executor(None, lambda: self.tokenizer.encode(execution_block, add_special_tokens=False))
                            if len(prompt_ids) + len(response_ids) + len(execution_ids) > self.max_model_len:
                                # If adding execution_ids exceeding max_model, we will not add execution_ids to response_ids
                                finish_reason = FINISH_REASON.EXCEED_LENGTH
                            else:
                                response_ids += execution_ids
                                response_mask += [0] * len(execution_ids)
                                if len(prompt_ids) + len(response_ids) == self.max_model_len:
                                    finish_reason = FINISH_REASON.EXCEED_LENGTH
                                else:
                                    finish_reason = FINISH_REASON.STOP

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

    @rollout_trace_op
    async def _run_val(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> TSValAgentLoopOutput:
        # We need to remove submission file first before doing inference
        workspace = data_instance["extra_info"]["workspace"]
        if workspace:
            submission_file = os.path.join(workspace, "submission.csv")
            if os.path.exists(submission_file):
                os.remove(submission_file)

        return await super()._run_val(messages, sampling_params, data_instance)