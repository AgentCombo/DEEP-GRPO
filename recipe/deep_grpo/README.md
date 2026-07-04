# DEEP-GRPO

DEEP-GRPO is a prefix-injection extension of GRPO for reasoning tasks. The core idea is to turn failed rollouts into new, teacher-verified hard states: when the student fails, a teacher model is asked to continue from the longest correct prefix, and the resulting prefix state is inserted back into training as an augmented prompt.

This recipe is implemented on top of verl's async vLLM rollout path. All method code, reward functions, and data preprocessing live in `recipe/deep_grpo/`. The current release focuses on the `forest` prefix pool, which keeps prompt-level tree structure and samples hard states with tree-balanced LRU.

## Method

1. Run standard GRPO rollouts from original prompts.
2. For each prompt group, collect failed trajectories whose reward is at or below `REWARD_THRESHOLD`.
3. Send failed trajectories to a background teacher worker.
4. The teacher generates a corrected solution. The agent loop finds the longest token prefix shared by the failed student response and the teacher response.
5. The teacher suffix is reward-verified. Low-reward or non-matching teacher outputs are discarded.
6. The forest inserts a child node whose state is the original prompt plus the verified correct prefix.
7. Later training batches sample active forest nodes using tree-level LRU and node-level LRU, so different original prompts stay balanced.
8. Each sampled forest node is treated as an augmented prompt. The student rolls out from that hard state and receives ordinary GRPO rewards and advantages.
9. If all rollouts from a node succeed, that node is deactivated. If they fail, the node can generate deeper teacher requests and the forest grows further.

The prefix-injection channel is on-policy from the injected state: the teacher chooses useful hard states and verifies correct prefixes, while the student still rolls out from those states with GRPO. This pure state curriculum is the released default — the base scripts train no teacher tokens at all. Optionally, the verified teacher suffix can also be trained on directly; the `sft`, `luffy`, and `snis` scripts are the corresponding ablation arms (see [Method Variants](#method-variants)).

## Experiments

DEEP-GRPO-1.5B trains Qwen2.5-Math-1.5B with the default MATH prefix-forest script (`Qwen2.5-Math-1.5B-MATH-compare-DEEPGRPO-prefix-forest.sh`, the pure state-curriculum arm — no teacher-token loss). Training and evaluation data follow the [Dr. GRPO repository](https://github.com/sail-sg/understand-r1-zero) setup. Accuracy (%) on five math benchmarks:

| Model | AIME24 | AMC | MATH500 | Minerva | Oly. | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-1.5B ([Yang et al., 2024](https://arxiv.org/abs/2409.12122)) | 16.7 | 43.4 | 61.8 | 15.1 | 28.4 | 33.1 |
| Qwen2.5-Math-1.5B-Instruct ([Yang et al., 2024](https://arxiv.org/abs/2409.12122)) | 10.0 | 48.2 | 74.2 | 26.5 | **40.2** | 39.8 |
| Oat-Zero-1.5B (Dr. GRPO) ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)) | 20.0 | 53.0 | 74.2 | 25.7 | 37.6 | 42.1 |
| **DEEP-GRPO-1.5B (ours)** | **23.3** | **54.2** | **77.0** | **29.0** | 37.0 | **44.1** |

## Installation

Use a Linux machine with NVIDIA GPUs. The scripts use FSDP + async vLLM rollout and assume CUDA is available.

The released scripts follow the original experiment setup: install the verl/vLLM training stack first, then install this repo without letting pip resolve and overwrite those pinned packages.

```bash
git clone https://github.com/AgentCombo/DEEP-GRPO.git
cd DEEP-GRPO

conda create -n deep-grpo python=3.10 -y
conda activate deep-grpo

# DEEP-GRPO uses the FSDP + vLLM path. Disable Megatron and SGLang unless you need them.
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .
pip install transformers==4.57.5
pip install latex2sympy2_extended
pip install math_verify
pip install scikit-learn
pip install openai
```

The teacher/judge model can run in a separate environment or on a separate machine. Serving Qwen3.5 requires a newer vLLM than the training stack pins, so keep the teacher server in its own environment:

```bash
conda create -n deep-grpo-teacher python=3.10 -y
conda activate deep-grpo-teacher
```

The install command depends on the NVIDIA **driver** on the teacher machine — check the `CUDA Version` reported by `nvidia-smi`. The vLLM wheels bundle the CUDA runtime, so no local CUDA Toolkit is needed, but the bundled runtime's major version must be supported by the driver:

- **Driver reports CUDA 13.x (R580 series or newer):** install the default wheels:

  ```bash
  pip install -U vllm
  ```

- **Driver reports CUDA 12.x (driver >= 525; e.g. driver 550 = CUDA 12.4):** CUDA 13 binaries do not run on 12.x drivers, so install the CUDA 12.9 wheel variant instead; it runs on any 12.x driver through CUDA minor-version compatibility. Pick the `+cu129` wheel for your vLLM version from the [GitHub release assets](https://github.com/vllm-project/vllm/releases). For vLLM 0.24.0:

  ```bash
  pip install uv
  uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
  ```

  Use `uv` rather than plain pip (uv gives the cu129 index priority over PyPI; pip would resolve `torch` back to the CUDA 13.0 default). If the environment already has a default `pip install -U vllm` in it, run `pip uninstall -y vllm torch torchvision torchaudio` first — leftover CUDA 13 packages satisfy the version pins and silently survive the reinstall. Verify: `python -c "import torch; print(torch.version.cuda)"` should print `12.9`.

- **Driver reports CUDA 11.x:** upgrade the driver before installing.

See the [Models](#models) section for the teacher launch command.

If your cluster uses a private PyPI mirror, add your mirror with `-i` or `--index-url`.

The launch scripts log to W&B by default. Either log in once:

```bash
wandb login
```

or run locally without uploading:

```bash
export WANDB_MODE=offline
```

## Data

Training scripts expect verl-style parquet files. By default:

```text
/data/hf-datasets/
  gsm8k/
    train.parquet
    test.parquet
  math/
    train.parquet
    test.parquet
  aime24/
    test.parquet
  minerva/
    test.parquet
  amc/
    test.parquet
  olympiad_bench/
    test.parquet
```

For the default GSM8K run, prepare data with:

```bash
export GSM8K_SAVE_PATH=/data/hf-datasets/gsm8k
python recipe/deep_grpo/data_preprocess/gsm8k.py
```

To use a local GSM8K mirror:

```bash
export GSM8K_DATA_SOURCE=/path/to/local/gsm8k
export GSM8K_SAVE_PATH=/data/hf-datasets/gsm8k
python recipe/deep_grpo/data_preprocess/gsm8k.py
```

For MATH-style experiments, prepare the MATH train/test parquet with:

```bash
export MATH_TRAIN_DATASET_PATH=/data/train/math_lvl3to5_8k
export MATH_TEST_DATASET_PATH=/data/evaluation_suite/math
export MATH_SAVE_PATH=/data/hf-datasets/math
python recipe/deep_grpo/data_preprocess/math_dataset.py
```

The MATH scripts also validate on AIME24, Minerva, AMC, and OlympiadBench (see `VAL_FILES` in the launch scripts), each with its own preprocessing script:

```bash
export AIME24_TEST_DATASET_PATH=/data/evaluation_suite/aime
export AIME24_SAVE_PATH=/data/hf-datasets/aime24
python recipe/deep_grpo/data_preprocess/aime24.py

export MINERVA_TEST_DATASET_PATH=/data/evaluation_suite/minerva
export MINERVA_SAVE_PATH=/data/hf-datasets/minerva
python recipe/deep_grpo/data_preprocess/minerva.py

export AMC_TEST_DATASET_PATH=/data/evaluation_suite/amc
export AMC_SAVE_PATH=/data/hf-datasets/amc
python recipe/deep_grpo/data_preprocess/amc.py

export OLYMPIAD_BENCH_TEST_DATASET_PATH=/data/evaluation_suite/olympiad_bench
export OLYMPIAD_BENCH_SAVE_PATH=/data/hf-datasets/olympiad_bench
python recipe/deep_grpo/data_preprocess/olympiadbench.py
```

These scripts read Hugging Face datasets saved to disk (`datasets.load_from_disk`) from the `*_TEST_DATASET_PATH` locations, so place your local copies of the evaluation sets there first. The MATH train set (`math_lvl3to5_8k`) and the evaluation suite (MATH500, AIME24, AMC, Minerva, OlympiadBench) come from the [Dr. GRPO repository](https://github.com/sail-sg/understand-r1-zero). Alternatively, override `VAL_FILES` on the launch script to validate on a subset (e.g. only `math/test.parquet`).

The DEEP-GRPO agent loop scores rollouts with the reward functions in `recipe/deep_grpo/reward/`. Keep the `data_source` fields produced by these scripts (`GSM8K`, `MATH`, `AIME24`, etc.) unless you also adjust the reward function.

## Models

The default GSM8K script uses:

```bash
MODEL=/data/hf-models/Qwen2.5-0.5B-Instruct
```

Example local download:

```bash
pip install modelscope
mkdir -p /data/hf-models
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct \
  --local_dir /data/hf-models/Qwen2.5-0.5B-Instruct
```

The MATH scripts default to:

```bash
MODEL=/data/hf-models/Qwen2.5-Math-1.5B
```

Example local download:

```bash
modelscope download --model Qwen/Qwen2.5-Math-1.5B \
  --local_dir /data/hf-models/Qwen2.5-Math-1.5B
```

Teacher annotation uses an OpenAI-compatible chat-completions endpoint. The released scripts default to `Qwen/Qwen3.5-27B-GPTQ-Int4` as the teacher. Start the teacher server first, then export the endpoint variables for the training run.

Example local download:

```bash
modelscope download --model Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --local_dir /data/hf-models/Qwen3.5-27B-GPTQ-Int4
```

Launch the server with vLLM in the teacher environment, following the official Qwen3.5 deployment guide. A single 80GB GPU is enough:

```bash
vllm serve /data/hf-models/Qwen3.5-27B-GPTQ-Int4 \
  --served-model-name Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 32768
```

Notes:

- Serving Qwen3.5 requires a recent vLLM (0.17 or newer; the Qwen team recommends the latest release). Older versions such as 0.10.x cannot load its hybrid GDN architecture.
- The released scripts set `TEACHER_ENABLE_THINKING=False` (sent per request via `chat_template_kwargs.enable_thinking`), so the teacher does not emit `<think>` blocks and `--reasoning-parser qwen3` is not required. Add it only if you enable thinking mode — otherwise the reasoning text would land in the response `content` and corrupt the longest-common-prefix match against the student's failed response.
- `--max-model-len 32768` is sized for this recipe, not for general Qwen3.5 serving: the student trains with a 1K-4K context, so each teacher request is a short failed trajectory plus the annotation template.
- With more GPUs, add `--tensor-parallel-size <n>` to serve more concurrent teacher requests.

Then point the training run at the endpoint:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY
export TEACHER_MODEL_NAME=Qwen/Qwen3.5-27B-GPTQ-Int4
```

Before starting a training run, verify the endpoint with curl. First check that the server is up and the served model name matches:

```bash
curl http://127.0.0.1:8001/v1/models
```

The response should list `"id": "Qwen/Qwen3.5-27B-GPTQ-Int4"`. Then send a chat completion shaped like a real teacher request — including `chat_template_kwargs.enable_thinking`, which the teacher client sends per request:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The reply's `choices[0].message.content` should contain the answer with no `<think>...</think>` block and no stray reasoning text before the answer.

`TEACHER_MODEL_NAME` must match the served model name. Any server compatible with the OpenAI chat API can be used instead of vLLM. The teacher must be strong enough to produce correct continuations for failed student trajectories.

## Run Experiments

Minimal GSM8K prefix-forest run (the default script is the pure state-curriculum arm):

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MODEL=/data/hf-models/Qwen2.5-0.5B-Instruct
export DATA_DIR=/data/hf-datasets
export OUTPUT_DIR=/data/deep-grpo-runs
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY
export TEACHER_MODEL_NAME=Qwen/Qwen3.5-27B-GPTQ-Int4

bash recipe/deep_grpo/scripts/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest.sh
```

Useful knobs (defaults shown are the GSM8K base script's):

```bash
# Run identity: names the W&B run, the log file, and the per-run output
# subdirectories. Default: auto-generated per script from the arm and key
# parameters, e.g. Qwen0.5B-GSM8K-prefix-forest-null-bs128-r0.125-<timestamp>.
export RUN_NAME=my-run
export PROJECT_NAME=DEEP-GRPO   # W&B project the runs are grouped under

# Experiment scale
export INJECTION_RATIO=0.125
export ROLLOUT_N=8
export PREFIX_ROLLOUT_N=8
export TRAIN_BATCH_SIZE=128
export PPO_MINI_BATCH_SIZE=16
export TOTAL_EPOCHS=100
export SAVE_FREQ=10
export TEST_FREQ=10
export SAVE_BEST_CHECKPOINT=True   # also keep the best-validation checkpoint

# GPU memory / throughput. The micro batch sizes only control gradient
# accumulation chunking — lower them to fit smaller GPUs without changing
# the effective batch size or training math.
export PPO_MICRO_BATCH_SIZE_PER_GPU=32          # forward+backward chunk per GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32     # log-prob forward chunk per GPU
export GPU_MEMORY_UTILIZATION=0.5               # vLLM rollout share of GPU memory
export ACTOR_PARAM_OFFLOAD=False                # offload FSDP params to CPU
export ACTOR_OPTIMIZER_OFFLOAD=False            # offload optimizer state to CPU
export MAX_PROMPT_LENGTH=512
export MAX_MODEL_LEN=1024                       # prompt + response token budget
```

Training and rollout share the same GPUs: vLLM reserves the `GPU_MEMORY_UTILIZATION` fraction and the FSDP actor uses the rest. If you hit CUDA OOM during the actor update, halve the two micro batch sizes first, then enable the offload flags, then lower `GPU_MEMORY_UTILIZATION`. If instead vLLM fails to allocate KV cache at startup, raise `GPU_MEMORY_UTILIZATION` or lower `MAX_MODEL_LEN`. The MATH/1.5B scripts default to a single 80GB GPU (`PPO_MICRO_BATCH_SIZE_PER_GPU=4`, `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8`, no offload); on 40GB cards drop to `2`/`4` and set `ACTOR_OPTIMIZER_OFFLOAD=True`.

MATH/1.5B SNIS arm (defaults sized for one 80GB training GPU; keep the teacher server on a different GPU):

```bash
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/data/hf-datasets
export OUTPUT_DIR=/data/deep-grpo-runs
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY
export TEACHER_MODEL_NAME=Qwen/Qwen3.5-27B-GPTQ-Int4

bash recipe/deep_grpo/scripts/Qwen2.5-Math-1.5B-MATH-compare-DEEPGRPO-prefix-forest-snis.sh
```

Teacher-token ablation arms (the default scripts train no teacher tokens; these add a teacher-token loss on top of the same state curriculum):

```bash
# Epoch-local suffix SFT (post-PPO NLL replay on teacher suffixes).
bash recipe/deep_grpo/scripts/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest-sft.sh

# LUFFY-style teacher-token loss.
bash recipe/deep_grpo/scripts/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest-luffy.sh

# SNIS teacher-token loss.
bash recipe/deep_grpo/scripts/Qwen2.5-0.5B-Instruct-GSM8K-compare-DEEPGRPO-prefix-forest-snis.sh
```

The same four arms exist for MATH/1.5B: `Qwen2.5-Math-1.5B-MATH-compare-DEEPGRPO-prefix-forest.sh` (default, pure state curriculum), plus the `-sft.sh`, `-luffy.sh`, and `-snis.sh` variants.

Logs are written to `$LOG_DIR/$RUN_NAME.log` (default `LOG_DIR=./logs`); checkpoints, validation dumps, and W&B files go to `$OUTPUT_DIR/checkpoints/$RUN_NAME`, `$OUTPUT_DIR/validation/$RUN_NAME`, and `$OUTPUT_DIR/wandb/$RUN_NAME`. All of these directories are created automatically.

Checkpoint retention: `trainer.max_actor_ckpt_to_keep` (the release scripts set 1) bounds how many `global_step_*` folders stay on disk. With `SAVE_BEST_CHECKPOINT=True` (the default), the trainer additionally saves a checkpoint whenever the validation score improves and keeps the best-scoring step alongside the newest one; the best step and score are recorded in `checkpoints/$RUN_NAME/best_checkpoint.json`. The ranking metric defaults to the cross-dataset average accuracy (`val-core/avg/...`) and can be overridden with `trainer.best_checkpoint_metric`.

## Method Variants

This recipe contains the full evolution of the method. All variants share the same rollout infrastructure and are selected purely by configuration; the released scripts run the newest **prefix-forest** form. Each step in the lineage exists because the previous one had a concrete failure mode, noted below.

### Lineage

**1. Branch expansion (earliest form).** This is the form described in the [DEEP-GRPO paper](https://arxiv.org/abs/2602.14169) (Deep Dense Exploration via pivot-driven resampling): pick branch points — *pivots* — on failed chains, then generate alternative continuations (branches) from those points in the same rollout pass. Branch points are picked either uniformly at random or by a learned utility model (`branching_strategy.py`, the paper's recoverability/depth utility function); responses are segmented into candidate points by `partition_strategy.py`. Main-chain and branch-chain tokens are trained with **separate losses** (the paper's dual-stream objective), weighted by `branch_chain_loss_lambda`. Key config:

```yaml
actor_rollout_ref.rollout.deep_grpo.expand_branch_chain: True
actor_rollout_ref.rollout.deep_grpo.pick_branch_chain_root_method: utility  # or random
actor_rollout_ref.rollout.deep_grpo.n_branch_points: 1
actor_rollout_ref.rollout.deep_grpo.branches_per_point: 8
actor_rollout_ref.rollout.deep_grpo.utility_sampling.*        # utility model knobs
actor_rollout_ref.actor.deep_grpo.branch_chain_loss_lambda: 1.0
```

`one_stage_mode=True` decouples this: branch points found at step *t* go into `branch_point_buffer.py` and are expanded at step *t+1*, in parallel with the next main-chain generation.

*Problems.* Generation cost is high: every selected branch point spawns `branches_per_point` extra rollouts on top of the main chains. And the model has no reliable signal for *where* to branch — random selection is blind, and the learned utility model is a heuristic proxy that itself needs training and tuning. Without knowing where the first real error is, most branches restart from states that are either already doomed or never wrong in the first place.

**2. Teacher suffix synthesis.** Fixes the branch-point-location problem: failed chains are sent to a background teacher (`teacher_worker.py`), and the teacher writes a corrected solution. The longest common prefix with the student's failed response locates the first error, and the teacher suffix is reward-verified before use. The verified result is consumed as **branch points**: entries are sampled from the teacher-annotated pool, the student generates `branches_per_entry` continuations from the error point during a later rollout pass (annotation is asynchronous, and entries are only sampled once the pool has accumulated `branch_batch_threshold` of them), and those rows are trained as branch/teacher chains in the same separate-loss structure as variant 1.

```yaml
actor_rollout_ref.rollout.deep_grpo.expand_branch_chain: False
actor_rollout_ref.rollout.deep_grpo.one_stage_mode: True
actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.enabled: True
```

*Problems.* Consumption is one-shot: a sampled entry leaves the pool and is never revisited, so an expensive teacher annotation buys exactly one training exposure — wasteful, and one exposure at a hard state is rarely enough for the student to actually learn it. There is also no notion of mastery: the trainer never checks whether the student can now solve the state it just practiced, so it can neither retire solved states nor keep drilling unsolved ones.

**3. Prefix injection (current form, released).** Fixes the reuse problem. The teacher pipeline is unchanged — what changes is what its output becomes. The verified correct prefix no longer spawns branch rows inside the current batch; it defines a persistent **hard state**: `original prompt + verified prefix` is stored in a pool and injected into later training batches as an ordinary prompt, mixed with dataset prompts at `injection_ratio`. The state is re-sampled across steps until the student masters it: all-success rollouts deactivate it, renewed failures send deeper teacher requests and grow it. There is no separate branch loss — rollouts from injected prompts flow through the **same unified GRPO loss** as dataset prompts, and training on teacher tokens themselves becomes an optional, orthogonal axis (next section) instead of being baked into the method.

Note the flag below: `teacher_suffix_synthesis.enabled: False` does **not** turn the teacher off. That flag enables variant 2's branch-point pipeline, which is mutually exclusive with `prefix_inject_mode` (both consume the same failed rollouts; the trainer asserts exactly one is active). Prefix injection runs its own teacher worker wired to the prefix pool, and it still reads the `teacher_suffix_synthesis.*` sub-keys (annotation concurrency, prefix-match thresholds) as shared teacher-client settings.

```yaml
actor_rollout_ref.rollout.deep_grpo.expand_branch_chain: False
actor_rollout_ref.rollout.deep_grpo.one_stage_mode: True
actor_rollout_ref.rollout.deep_grpo.teacher_suffix_synthesis.enabled: False
actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.enabled: True
actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.pool_type: forest   # flat | chain | forest
actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode.injection_ratio: 0.125
```

### Pool backends: flat → chain → forest

The hard-state pool itself went through three generations, selected by `prefix_inject_mode.pool_type`. `flat` and `chain` are kept for ablations but not exposed as release scripts.

- **`flat`** (`synthetic_prompt_pool.py`): a structureless bag of injected prompts, sampled by a variance weight on each state's last observed success rate: `w = 4p(1-p)`, with floors `w_floor_hard=0.2` for all-fail states and `w_floor_mastered=0.01` for solved ones. Two problems. The weight peaks at `p=0.5`, so half-solved states get scanned the most while the hardest states are starved — `p=0.1` gets `w=0.36` and a still-all-fail state only `0.2`, yet those are exactly the states with the most to teach. And there is no per-original-prompt balancing: prompts that spawn many entries dominate the sample, crowding out the rest.
- **`chain`** (`prefix_chain_pool.py`): one deepening chain of prefixes per original prompt. This fixes per-prompt balance but commits to a single path: if the shallow prefix is a poor fix, every deeper state extends it, and all later exploration on that prompt is biased by the early commitment. There is no way to back out and try an alternative prefix at the same depth.
- **`forest`** (`prefix_forest_pool.py`, released): keeps a tree per original prompt, so multiple alternative prefixes stay alive at every depth. It also drops success-rate weighting entirely and samples the way a dataloader scans a dataset: tree-level LRU picks the least-recently-visited prompt, node-level LRU the least-recently-visited state within it. Every live hard state gets revisited at a uniform rate regardless of its current success rate; mastery is handled by deactivation, not down-weighting.

### Teacher-token loss variants (orthogonal axis)

On top of prefix injection, the verified teacher suffix can optionally be trained on directly. `actor.deep_grpo.teacher_loss_type` selects the arm:

| `TEACHER_LOSS_TYPE` | Meaning | Script |
| --- | --- | --- |
| `null` | No teacher-token loss: pure state curriculum. | default scripts |
| `sft` | NLL on teacher tokens, replayed epoch-locally post-PPO. | `*-sft.sh` |
| `luffy` | LUFFY-style policy shaping on teacher tokens. | `*-luffy.sh` |
| `snis` | Self-normalized importance-weighted NLL on teacher tokens. | `*-snis.sh` |

In our experiments, none of the three teacher-token losses (epoch-local SFT, LUFFY, SNIS) delivered a consistent improvement over the pure state curriculum (`null`): the gains come from rolling out on the injected hard states, not from imitating teacher tokens. That is why `null` is the released default; the other arms are kept so the ablation is reproducible.

## Important Parameters

| Parameter | Meaning |
| --- | --- |
| `INJECTION_RATIO` | Fraction of each training batch filled with forest-injected prefix prompts. |
| `PREFIX_ROLLOUT_N` | Number of student rollouts from each injected prefix prompt. Defaults to `ROLLOUT_N`. |
| `REWARD_THRESHOLD` | Rollouts with reward `<=` this value are considered failures for teacher annotation. |
| `TEACHER_MAX_CONCURRENT` | Max concurrent teacher annotation requests. |
| `MIN_PREFIX_MATCH_TOKENS` | Minimum token overlap required between failed student response and teacher solution. |
| `MIN_PREFIX_MATCH_RATIO` | Minimum overlap ratio required for teacher prefix matching. |
| `MIN_SUFFIX_LEN` | Minimum teacher suffix length after prefix matching. |
| `SUFFIX_SFT_ENABLED` | Enables epoch-local teacher suffix SFT replay (the `*-sft.sh` ablation arm). Default False. |
| `PREFIX_TEACHER_CONTINUATION_ENABLED` | Enables the in-batch teacher-continuation path; the default (none), `*-luffy.sh`, and `*-snis.sh` arms all set it to True. With `TEACHER_LOSS_TYPE=luffy/snis` it appends each injected group's verified teacher suffix as an extra teacher row trained jointly with the student rollouts; with `null` it applies the same student-side advantage handling but builds no teacher row, giving the clean state-curriculum-only default. Mutually exclusive with `SUFFIX_SFT_ENABLED`. |
| `TEACHER_LOSS_TYPE` | `null`, `luffy`, `snis`, or `sft`, depending on the experiment arm. |

## Code Layout

```text
recipe/deep_grpo/
├── scripts/                  # launch scripts for GSM8K and MATH experiments
├── agent_loop/               # rollout logic
│   ├── tree_search_agent_loop.py    # TSAgentLoop: generic tree-search core (method-agnostic)
│   ├── deep_grpo_agent_loop.py      # DeepGRPOAgentLoop(TSAgentLoop): all DEEP-GRPO variants
│   ├── reasoning_agent_loop.py      # math-reasoning loop used by the released scripts
│   ├── treerl_agent_loop.py         # TreeRL comparison baseline
│   ├── deep_analyze_agent_loop.py   # data-analysis agent (code execution)
│   ├── search_agent_loop.py         # retrieval-augmented agent (multi-hop QA)
│   ├── outputs.py                   # train/val output types (extend verl AgentLoopOutput)
│   ├── code_executor.py             # sandboxed code execution for deep-analyze
│   └── retriever.py                 # HTTP retriever client for search agent
├── pools/                    # state pools and buffers
│   ├── prefix_forest_pool.py        # the released hard-state forest (pool_type=forest)
│   ├── prefix_chain_pool.py         # earlier chain backend
│   ├── synthetic_prompt_pool.py     # earlier flat backend
│   ├── failed_trajectory_pool.py    # failed rollouts awaiting teacher annotation
│   ├── teacher_annotated_pool.py    # teacher-verified entries ready for use
│   └── branch_point_buffer.py       # branch-point FIFO (one-stage branch expansion)
├── reward/                   # rule-based and LLM-judge reward functions per data source
│   └── prompts/                     # judge prompt templates
├── data_preprocess/          # dataset -> verl parquet converters (GSM8K, MATH, ...)
├── tests/                    # standalone unit tests (run without GPUs)
├── protocol.py               # core data structures (Node, rewards, teacher suffixes, ...)
├── prompts.py                # teacher selection and suffix-synthesis templates
├── teacher_worker.py         # background teacher annotation worker
├── teacher_suffix_utils.py   # teacher suffix post-processing
├── utils.py                  # teacher-endpoint client with retry
├── branching_strategy.py     # branch-point selection (random / utility)
└── partition_strategy.py     # response segmentation (sentence / token count / fixed)
```

`TSAgentLoop` is the method-agnostic tree-search rollout engine. `DeepGRPOAgentLoop` subclasses it with every DEEP-GRPO variant, and all concrete task loops (reasoning, TreeRL, deep-analyze, search) subclass `DeepGRPOAgentLoop`.

## Debugging

Run the standalone unit tests locally (no GPU or verl stack needed for the pool tests):

```bash
PYTHONPATH=. python recipe/deep_grpo/tests/test_prefix_forest_pool.py
PYTHONPATH=. python recipe/deep_grpo/tests/test_prefix_chain_pool.py
PYTHONPATH=. python recipe/deep_grpo/tests/test_teacher_suffix_eos.py
```

Enable prefix dumps for inspection:

```bash
export PREFIX_DEBUG_DUMP_ENABLED=True
export PREFIX_DEBUG_DUMP_FREQ=10
export PREFIX_DEBUG_DUMP_MAX_PREFIXES=16
```

Useful metrics include:

- `forest_pool/num_trees`
- `forest_pool/active_nodes`
- `forest_pool/events_added_this_step`
- `forest_pool/nodes_retired_this_step`
- `teacher_suffix/*`
- `prefix_suffix_sft/*`
- `prefix_luffy/*`

## Citation

The paper introduces Deep Dense Exploration (DDE) and instantiates it as DEEP-GRPO in its pivot-driven resampling form — the branch-expansion lineage (variant 1 in [Method Variants](#method-variants)) with the utility-based pivot selection and dual-stream objective. The prefix-forest method released here is newer, ongoing work built on the same infrastructure.

```bibtex
@misc{guo2026deepgrpo,
  title         = {Deep Dense Exploration for LLM Reinforcement Learning via Pivot-Driven Resampling},
  author        = {Yiran Guo and Zhongjian Qiao and Yingqi Xie and Jie Liu and Dan Ye and Ruiqing Zhang and Shuang Qiu and Lijie Xu},
  year          = {2026},
  eprint        = {2602.14169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2602.14169}
}
```
