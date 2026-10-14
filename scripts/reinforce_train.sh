#!/usr/bin/env bash
# GPU RL training with OpenRLHF + Ray + vLLM. Start the services first (each in its own shell):
#   python -m kghopper.kg.server --config configs/paper_gpu.yaml --port 5005          # KG tool
#   python reward-remote/reward_server.py --config configs/paper_gpu.yaml --data_path $DATA_PATH --port 1278
#   bash scripts/ray_start.sh
# then: bash scripts/reinforce_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="$PWD:$PWD/OpenRLHF-RAG:${PYTHONPATH:-}"

DATA_PATH=${DATA_PATH:-data/train/rl.jsonl}             # 2,000 RL prompts (python -m kghopper.prepare_data rl; docs/DATA.md)
PRETRAIN=${PRETRAIN:-checkpoints/sft}                    # cold-start SFT model of the backbone (Qwen2.5-7B or Llama-3.1-8B-Instruct)
SAVE_MODEL_NAME=${SAVE_MODEL_NAME:-kg-hopper-grpo}
SAVE_DIR=${SAVE_DIR:-checkpoints/openrlhf}
KG_URL=${KG_URL:-http://127.0.0.1:5005}
RM_PORT=${RM_PORT:-1278}

# RL settings of the paper (Sec. III-D, IV-B); override any of them through the environment.
ADV=${ADV:-group_norm}            # GRPO: group-normalised advantages
N_SAMPLES=${N_SAMPLES:-16}        # 16 rollouts per query
RBS=${RBS:-16}                    # batch size 16 (prompts per rollout batch)
TBS=${TBS:-256}                   # samples per update = RBS x N_SAMPLES (one on-policy update per batch)
EPISODE=${EPISODE:-2}             # 2 epochs over the RL prompts
LR=${LR:-1e-6}                    # learning rate 1e-6
KL=${KL:-1e-5}                    # KL penalty 1e-5, as a GRPO-style k3 KL loss (--use_kl_loss)
EPS_CLIP=${EPS_CLIP:-0.2}         # PPO clip ratio 0.2
TEMP=${TEMP:-1.0}                 # rollout temperature 1
HISTORY_RESAMPLING_EPISODE=${HISTORY_RESAMPLING_EPISODE:-1}  # from epoch 2 (0-based 1), one-hop questions are replaced
# Not fixed by the paper:
MAX_TURNS=${MAX_TURNS:-8}               # generation rounds per trajectory
TOKENS_PER_TURN=${TOKENS_PER_TURN:-512} # new tokens per round
GEN_MAX_LEN=${GEN_MAX_LEN:-29000}       # total response budget across rounds

mkdir -p "$SAVE_DIR/$SAVE_MODEL_NAME" logs
# The Ray job may not run in this directory: make existing relative paths absolute (HF ids stay as they are).
absolute() { if [ -e "$1" ]; then (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); else echo "$1"; fi; }
DATA_PATH=$(absolute "$DATA_PATH"); PRETRAIN=$(absolute "$PRETRAIN"); SAVE_DIR=$(absolute "$SAVE_DIR")

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(--use_wandb "$WANDB_API_KEY" --wandb_run_name "$SAVE_MODEL_NAME")
fi

ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH\"}}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain "$PRETRAIN" \
   --remote_rm_url "http://localhost:${RM_PORT}/get_reward" \
   --kg_url "$KG_URL" \
   --max_turns "$MAX_TURNS" \
   --max_new_tokens_per_turn "$TOKENS_PER_TURN" \
   --save_path "$SAVE_DIR/$SAVE_MODEL_NAME" \
   --ckpt_path "$SAVE_DIR/$SAVE_MODEL_NAME" \
   --micro_train_batch_size 1 \
   --train_batch_size "$TBS" \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size "$RBS" \
   --advantage_estimator "$ADV" \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes "$EPISODE" \
   --lr_warmup_ratio 0.0 \
   --n_samples_per_prompt "$N_SAMPLES" \
   --prompt_max_len 1024 \
   --generate_max_len "$GEN_MAX_LEN" \
   --zero_stage 2 \
   --bf16 \
   --load_checkpoint \
   --actor_learning_rate "$LR" \
   --critic_learning_rate 9e-6 \
   --init_kl_coef "$KL" \
   --use_kl_loss \
   --use_kl_estimator_k3 \
   --eps_clip "$EPS_CLIP" \
   --history_resampling_episode "$HISTORY_RESAMPLING_EPISODE" \
   --prompt_data "$DATA_PATH" \
   --input_key question \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 3 \
   --temperature "$TEMP" \
   --overlap_comm \
   --packing_samples \
   ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
