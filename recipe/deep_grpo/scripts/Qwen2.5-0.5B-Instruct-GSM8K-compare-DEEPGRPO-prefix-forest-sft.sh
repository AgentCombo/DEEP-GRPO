#!/usr/bin/env bash

# Epoch-local suffix-SFT ablation arm on GSM8K.
# Same forest state curriculum as the default (none) script; additionally, the
# verified teacher suffixes collected in each PPO epoch are replayed with a
# post-PPO NLL (SFT) phase on a separate teacher optimizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUFFIX_SFT_ENABLED=${SUFFIX_SFT_ENABLED:-True}
export SUFFIX_SFT_SCHEDULE=${SUFFIX_SFT_SCHEDULE:-epoch_local}
export PREFIX_TEACHER_CONTINUATION_ENABLED=${PREFIX_TEACHER_CONTINUATION_ENABLED:-False}
export TEACHER_LOSS_TYPE=${TEACHER_LOSS_TYPE:-sft}
export TEACHER_UPDATE_MODE=${TEACHER_UPDATE_MODE:-post_ppo}
export TEACHER_LOSS_COEF=${TEACHER_LOSS_COEF:-1.0}
export TEACHER_LOSS_REDUCTION=${TEACHER_LOSS_REDUCTION:-separate_stream_mean}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME=${RUN_NAME:-Qwen0.5B-GSM8K-pforest-SFT-${SUFFIX_SFT_SCHEDULE}-r${INJECTION_RATIO:-0.125}-${TIMESTAMP}}

exec bash "$SCRIPT_DIR/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest.sh"
