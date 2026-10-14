#!/usr/bin/env bash
# Start the Ray head node used by scripts/reinforce_train.sh.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
HEAD_NODE_IP=${HEAD_NODE_IP:-127.0.0.1}
NUM_GPUS=${NUM_GPUS:-8}

ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus "${NUM_GPUS}" --port 8266 --dashboard-port 8267
