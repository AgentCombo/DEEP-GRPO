#!/usr/bin/env bash

# SNIS arm on Qwen2.5-Math-1.5B + MATH (harder task, larger model).
#
# Purpose: GSM8K-0.5B is an "oasis" — the student can mostly rollout the correct
# answer itself, so the teacher term stays sparse (token fraction ~0.16%) and
# SNIS ties pure-inject on test. This config probes whether MATH/1.5B is a
# genuine "desert": if the teacher signal stays sparse here too, SNIS will tie
# pure-inject again; if it becomes dense (many nodes the student truly cannot
# solve, teacher rarely retires), this is the regime where the SNIS escape
# theorem has something to escape.
#
# DIAGNOSTIC FIRST: before committing to a full three-arm comparison, run this
# and read prefix_luffy/teacher_token_fraction and the ratio
# teacher_not_better_skipped : teacher_rows_built. Dense teacher signal => SNIS
# worth the full ablation; sparse => pure-inject likely suffices.

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

# ---- SNIS teacher continuation (the method under test) ----
export SUFFIX_SFT_ENABLED=${SUFFIX_SFT_ENABLED:-False}
export PREFIX_TEACHER_CONTINUATION_ENABLED=${PREFIX_TEACHER_CONTINUATION_ENABLED:-True}
export TEACHER_LOSS_TYPE=${TEACHER_LOSS_TYPE:-snis}
export TEACHER_UPDATE_MODE=${TEACHER_UPDATE_MODE:-joint}
export TEACHER_LOSS_COEF=${TEACHER_LOSS_COEF:-1.0}
export TEACHER_LOSS_REDUCTION=${TEACHER_LOSS_REDUCTION:-mixed_token_mean}
export SNIS_BETA=${SNIS_BETA:-1.0}
# MATH is a harder "desert" than GSM8K: inject more prefix nodes than the
# GSM8K default (0.125). Higher r = more hard-node practice + denser teacher
# signal, at the cost of fewer from-scratch original rollouts. Sweep candidate.
export INJECTION_RATIO=${INJECTION_RATIO:-0.25}

# ---- Teacher annotation backend ----
# Intentionally NOT set here: inherited from the base script defaults and/or
# whatever you export in your shell — exactly as the GSM8K snis launcher does.
# (The base script defaults OPENAI_API_KEY / OPENAI_BASE_URL / TEACHER_MODEL_NAME;
#  override them in your shell the same way you did for the GSM8K run.)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME=${RUN_NAME:-Qwen1.5B-MATH-pforest-SNIS-b${SNIS_BETA}-r${INJECTION_RATIO:-0.125}-${TIMESTAMP}}

exec bash "$SCRIPT_DIR/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest.sh"
