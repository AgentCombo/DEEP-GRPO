#!/usr/bin/env bash

# SNIS arm of the three-arm ablation (none / luffy / snis).
# Teacher continuation rows trained with self-normalized weighted NLL:
#   L_teacher = -sg[w̃]·logπ,  w̃ = exp(A*/β) / mean_j exp(A_j/β)
# over the mixed group (students + teacher). See actor.yaml snis_beta docs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUFFIX_SFT_ENABLED=${SUFFIX_SFT_ENABLED:-False}
export PREFIX_TEACHER_CONTINUATION_ENABLED=${PREFIX_TEACHER_CONTINUATION_ENABLED:-True}
export TEACHER_LOSS_TYPE=${TEACHER_LOSS_TYPE:-snis}
export TEACHER_UPDATE_MODE=${TEACHER_UPDATE_MODE:-joint}
export TEACHER_LOSS_COEF=${TEACHER_LOSS_COEF:-1.0}
export TEACHER_LOSS_REDUCTION=${TEACHER_LOSS_REDUCTION:-mixed_token_mean}
export SNIS_BETA=${SNIS_BETA:-1.0}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME=${RUN_NAME:-Qwen0.5B-GSM8K-pforest-SNIS-b${SNIS_BETA}-r${INJECTION_RATIO:-0.125}-${TIMESTAMP}}

exec bash "$SCRIPT_DIR/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest.sh"
