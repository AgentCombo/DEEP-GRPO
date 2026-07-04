"""DEEP-GRPO agent loops.

- `tree_search_agent_loop.TSAgentLoop`: generic tree-search rollout core
  (step-wise chain rollout, parallel expansion, compression, scoring,
  group-baseline output packing). Method-agnostic.
- `deep_grpo_agent_loop.DeepGRPOAgentLoop`: extends TSAgentLoop with every
  DEEP-GRPO variant (branch expansion, teacher suffix synthesis, prefix
  injection) without modifying the core.
- `reasoning_agent_loop.ReasoningAgentLoop`: math-reasoning loop used by
  the released scripts.
- `treerl_agent_loop.TreeRLAgentLoop`: TreeRL comparison baseline.
- `deep_analyze_agent_loop` / `search_agent_loop`: task-specific loops for
  data-analysis and retrieval-augmented experiments.
"""
