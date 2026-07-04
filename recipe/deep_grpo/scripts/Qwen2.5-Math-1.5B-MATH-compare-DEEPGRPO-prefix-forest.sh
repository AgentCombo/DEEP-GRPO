#!/usr/bin/env bash

# Default arm on Qwen2.5-Math-1.5B + MATH: pure state curriculum ("none").
# The forest injects teacher-verified prefix states and the student rolls out
# from them with plain GRPO (student-only advantage baseline). No teacher token
# enters the loss — the teacher-token arms are the -sft / -luffy / -snis
# wrappers next to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Model / data (MATH, 1.5B) ----
export MODEL=${MODEL:-/data/hf-models/Qwen2.5-Math-1.5B}
export DATA_DIR=${DATA_DIR:-/data/hf-datasets}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/math/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/math/test.parquet,$DATA_DIR/aime24/test.parquet,$DATA_DIR/minerva/test.parquet,$DATA_DIR/amc/test.parquet,$DATA_DIR/olympiad_bench/test.parquet]"}

# ---- Sequence lengths (MATH needs long CoT) ----
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# ---- 1.5B memory, sized for a single 80GB A100 (keep the teacher server on
# a different GPU). On 40GB cards fall back to the original values:
# PPO_MICRO_BATCH_SIZE_PER_GPU=2, LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4,
# ACTOR_OPTIMIZER_OFFLOAD=True. ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}

# ---- Teacher-token arm: none (pure state curriculum, base script defaults) ----
# SUFFIX_SFT_ENABLED=False, PREFIX_TEACHER_CONTINUATION_ENABLED=True,
# TEACHER_LOSS_TYPE=null, TEACHER_UPDATE_MODE=joint, and
# TEACHER_LOSS_REDUCTION=mixed_token_mean all come from the base script.
# MATH is a harder "desert" than GSM8K: inject more prefix nodes than the
# GSM8K default (0.125). Keep the ratio identical across the four MATH arms so
# comparisons isolate the teacher-token loss, not the injection amount.
export INJECTION_RATIO=${INJECTION_RATIO:-0.25}

# ---- Teacher annotation backend ----
# Inherited from base script defaults / your shell exports (same as GSM8K).

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME=${RUN_NAME:-Qwen1.5B-MATH-pforest-none-r${INJECTION_RATIO:-0.125}-${TIMESTAMP}}

exec bash "$SCRIPT_DIR/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest.sh"
