#!/usr/bin/env bash

set -euo pipefail

# Prefix-forest inject experiment for GSM8K.
#
# Default arm: pure state curriculum ("none") — the forest injects verified
# prefix states and the student rolls out from them with plain GRPO; no
# teacher-token loss. The -sft / -luffy / -snis wrappers override the
# teacher-token exports below to select the other ablation arms.
#
# Compared with Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-one-stage.sh:
#   - one_stage_mode remains enabled
#   - buffer branch expansion is disabled
#   - prefix_inject_mode.enabled=True
#   - prefix_inject_mode.pool_type=forest
#   - teacher_suffix_synthesis.enabled=False, but its sub-config is reused by
#     the forest teacher worker for annotation concurrency / prefix matching.

export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export CUDA_VISIBLE_DEVICES

MODEL=${MODEL:-/data/hf-models/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-/data/hf-datasets}
OUTPUT_DIR=${OUTPUT_DIR:-/data}
INJECTION_RATIO=${INJECTION_RATIO:-0.125}
PAIRED_EVAL_ENABLED=${PAIRED_EVAL_ENABLED:-False}
PAIRED_EVAL_FREQ=${PAIRED_EVAL_FREQ:-10}
PAIRED_EVAL_NUM_PAIRS=${PAIRED_EVAL_NUM_PAIRS:-4}
PREFIX_DEBUG_DUMP_ENABLED=${PREFIX_DEBUG_DUMP_ENABLED:-False}
PREFIX_DEBUG_DUMP_DIR=${PREFIX_DEBUG_DUMP_DIR:-null}
PREFIX_DEBUG_DUMP_FREQ=${PREFIX_DEBUG_DUMP_FREQ:-10}
PREFIX_DEBUG_DUMP_MAX_PREFIXES=${PREFIX_DEBUG_DUMP_MAX_PREFIXES:-16}
PREFIX_DEBUG_DUMP_MAX_DEEPER_EXAMPLES=${PREFIX_DEBUG_DUMP_MAX_DEEPER_EXAMPLES:-4}
PREFIX_DEBUG_DUMP_MAX_ROLLOUTS_PER_PREFIX=${PREFIX_DEBUG_DUMP_MAX_ROLLOUTS_PER_PREFIX:-8}
PREFIX_DEBUG_DUMP_MAX_TEXT_CHARS=${PREFIX_DEBUG_DUMP_MAX_TEXT_CHARS:-4000}
PREFIX_DEBUG_DUMP_FULL_TEXT=${PREFIX_DEBUG_DUMP_FULL_TEXT:-False}
SUFFIX_SFT_ENABLED=${SUFFIX_SFT_ENABLED:-False}
SUFFIX_SFT_SCHEDULE=${SUFFIX_SFT_SCHEDULE:-epoch_local}
SUFFIX_SFT_EPOCH_PASSES=${SUFFIX_SFT_EPOCH_PASSES:-5}
SUFFIX_SFT_MAX_BATCHES_PER_STEP=${SUFFIX_SFT_MAX_BATCHES_PER_STEP:-1}
SUFFIX_SFT_MAX_NODES_PER_TREE_PER_STEP=${SUFFIX_SFT_MAX_NODES_PER_TREE_PER_STEP:-1}
TEACHER_SFT_LR=${TEACHER_SFT_LR:-1e-6}
TEACHER_MINI_BATCH_SIZE=${TEACHER_MINI_BATCH_SIZE:-64}
TEACHER_ALLOW_PARTIAL_BATCH=${TEACHER_ALLOW_PARTIAL_BATCH:-False}
TEACHER_LOSS_TYPE=${TEACHER_LOSS_TYPE:-null}
TEACHER_UPDATE_MODE=${TEACHER_UPDATE_MODE:-joint}
TEACHER_LOSS_COEF=${TEACHER_LOSS_COEF:-1.0}
TEACHER_LOSS_REDUCTION=${TEACHER_LOSS_REDUCTION:-mixed_token_mean}
LUFFY_GAMMA=${LUFFY_GAMMA:-0.1}
SNIS_BETA=${SNIS_BETA:-1.0}
PREFIX_TEACHER_CONTINUATION_ENABLED=${PREFIX_TEACHER_CONTINUATION_ENABLED:-True}
TEACHER_MAX_CONCURRENT=${TEACHER_MAX_CONCURRENT:-32}
TEACHER_POLL_INTERVAL=${TEACHER_POLL_INTERVAL:-2.0}
MIN_PREFIX_MATCH_TOKENS=${MIN_PREFIX_MATCH_TOKENS:-10}
MIN_PREFIX_MATCH_RATIO=${MIN_PREFIX_MATCH_RATIO:-0.05}
MIN_SUFFIX_LEN=${MIN_SUFFIX_LEN:-100}
REWARD_THRESHOLD=${REWARD_THRESHOLD:-0.0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
PREFIX_ROLLOUT_N=${PREFIX_ROLLOUT_N:-$ROLLOUT_N}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT:-True}
# Dataset + sequence-length knobs (overridable so the same script serves
# GSM8K/0.5B and MATH/1.5B). Defaults preserve the GSM8K-0.5B behavior.
TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/gsm8k/train.parquet]"}
VAL_FILES=${VAL_FILES:-"[$DATA_DIR/gsm8k/test.parquet]"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
if [[ -z "${N_GPUS_PER_NODE:-}" ]]; then
    IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
    N_GPUS_PER_NODE=${#CUDA_DEVICE_LIST[@]}
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-Qwen0.5B-GSM8K-prefix-forest-${TEACHER_LOSS_TYPE}-bs${TRAIN_BATCH_SIZE}-r${INJECTION_RATIO}-${TIMESTAMP}}
export WANDB_DIR=$OUTPUT_DIR/wandb/$RUN_NAME

export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}
export TEACHER_MODEL_NAME=${TEACHER_MODEL_NAME:-Qwen/Qwen3.5-27B-GPTQ-Int4}
export TEACHER_ENABLE_THINKING=${TEACHER_ENABLE_THINKING:-False}

export RAY_local_fs_capacity_threshold=1

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR" "$WANDB_DIR"

# Keep xtrace below secret exports so OPENAI_API_KEY is not printed.
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.deep_grpo.teacher_loss_type=$TEACHER_LOSS_TYPE \
    actor_rollout_ref.actor.deep_grpo.teacher_update_mode=$TEACHER_UPDATE_MODE \
    actor_rollout_ref.actor.deep_grpo.teacher_loss_coef=$TEACHER_LOSS_COEF \
    actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction=$TEACHER_LOSS_REDUCTION \
    actor_rollout_ref.actor.deep_grpo.luffy_gamma=$LUFFY_GAMMA \
    actor_rollout_ref.actor.deep_grpo.snis_beta=$SNIS_BETA \
    actor_rollout_ref.actor.deep_grpo.teacher_sft_optim.lr=$TEACHER_SFT_LR \
    actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size=$TEACHER_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.deep_grpo.teacher_allow_partial_batch=$TEACHER_ALLOW_PARTIAL_BATCH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.deep_grpo.default_agent_name=reasoning_agent_loop \
    actor_rollout_ref.rollout.deep_grpo.one_stage_mode=True \
    actor_rollout_ref.rollout.deep_grpo.expand_branch_chain=False \
    actor_rollout_ref.rollout.deep_grpo.low_quality_trajectory_reward_threshold=$REWARD_THRESHOLD \
    actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.enabled=False \
    actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.max_concurrent_annotations=$TEACHER_MAX_CONCURRENT \
    actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.min_prefix_match_tokens=$MIN_PREFIX_MATCH_TOKENS \
    actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.min_prefix_match_ratio=$MIN_PREFIX_MATCH_RATIO \
    +actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.min_suffix_len=$MIN_SUFFIX_LEN \
    actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.poll_interval=$TEACHER_POLL_INTERVAL \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.enabled=True \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.pool_type=forest \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.injection_ratio=$INJECTION_RATIO \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.rollout_n=$PREFIX_ROLLOUT_N \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.suffix_sft_maturation.enabled=$SUFFIX_SFT_ENABLED \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.suffix_sft_maturation.schedule=$SUFFIX_SFT_SCHEDULE \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.suffix_sft_maturation.epoch_passes=$SUFFIX_SFT_EPOCH_PASSES \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.suffix_sft_maturation.max_batches_per_step=$SUFFIX_SFT_MAX_BATCHES_PER_STEP \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.suffix_sft_maturation.max_nodes_per_tree_per_step=$SUFFIX_SFT_MAX_NODES_PER_TREE_PER_STEP \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.teacher_continuation.enabled=$PREFIX_TEACHER_CONTINUATION_ENABLED \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.paired_eval.enabled=$PAIRED_EVAL_ENABLED \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.paired_eval.freq=$PAIRED_EVAL_FREQ \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.paired_eval.num_pairs=$PAIRED_EVAL_NUM_PAIRS \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.enabled=$PREFIX_DEBUG_DUMP_ENABLED \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.dir=$PREFIX_DEBUG_DUMP_DIR \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.freq=$PREFIX_DEBUG_DUMP_FREQ \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.max_prefixes=$PREFIX_DEBUG_DUMP_MAX_PREFIXES \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.max_deeper_examples=$PREFIX_DEBUG_DUMP_MAX_DEEPER_EXAMPLES \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.max_rollouts_per_prefix=$PREFIX_DEBUG_DUMP_MAX_ROLLOUTS_PER_PREFIX \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.max_text_chars=$PREFIX_DEBUG_DUMP_MAX_TEXT_CHARS \
    actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.debug_dump.full_text=$PREFIX_DEBUG_DUMP_FULL_TEXT \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.whiten_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME:-DEEP-GRPO} \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_best_checkpoint=$SAVE_BEST_CHECKPOINT \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$OUTPUT_DIR/validation/$RUN_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/$RUN_NAME \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    ray_init.num_cpus=64 \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a "$LOG_DIR/$RUN_NAME.log"
