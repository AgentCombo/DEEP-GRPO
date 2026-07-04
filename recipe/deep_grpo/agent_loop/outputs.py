"""Agent-loop output types.

These extend verl's AgentLoopOutput (pydantic), so they live in the
agent_loop package rather than protocol.py to keep the latter import-light
for the pool unit tests.
"""

from typing import List, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

from recipe.deep_grpo.protocol import RewardInfo


class TSValAgentLoopOutput(AgentLoopOutput):
    reward: float
    reward_info: Optional[RewardInfo] = None


class TSTrainAgentLoopOutput(TSValAgentLoopOutput):
    tree_id: str
    node_id: str
    advantage: float
    token_level_advantages: Optional[List[float]] = None
