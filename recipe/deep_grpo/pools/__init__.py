"""DEEP-GRPO state pools and buffers.

Queues and hard-state stores between rollout, the background teacher
worker, and the trainer:

- `failed_trajectory_pool`: failed rollouts awaiting teacher annotation.
- `teacher_annotated_pool`: teacher-verified branch entries ready for use.
- `prefix_forest_pool`: the released hard-state forest (prefix_inject_mode
  pool_type=forest).
- `prefix_chain_pool` / `synthetic_prompt_pool`: earlier chain / flat pool
  backends.
- `branch_point_buffer`: branch-point FIFO for the one-stage
  branch-expansion variant.
"""
