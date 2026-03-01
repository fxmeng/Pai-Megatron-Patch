#!/bin/bash
CURRENT_DIR=/workspace/rumimeng/Youtu-Megatron/
cd ${CURRENT_DIR}
export PYTHONPATH=${CURRENT_DIR}:${CURRENT_DIR}/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
apt-get update && apt-get install -y iproute2
# ================== 环境变量设置 ==================
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export GLOO_SOCKET_IFNAME=$(ip addr | grep -i enp | head -1 | awk '{print $2}' | tr -d ':')
export NCCL_SOCKET_IFNAME=$(ip addr | grep -i enp | head -1 | awk '{print $2}' | tr -d ':')

echo $NNODES
echo $NPROC_PER_NODE
echo $CUDA_DEVICE_MAX_CONNECTIONS
echo $GLOO_SOCKET_IFNAME
echo $NCCL_SOCKET_IFNAME
echo $PYTHONPATH

TP=1
PP=1
EP=8
ETP=1
CP=1
MBS=1
GBS=80
SEQ_LEN=25600
TRAIN_DATA_PATH=/workspace/data_02111332/vl_wds/HuggingFaceM4/
VALID_DATA_PATH=/workspace/data_02111332/vl_wds/HuggingFaceM4/
PRETRAIN_CHECKPOINT_PATH=/workspace/rumimeng/huggingface/Qwen/Qwen3-VL-5B-A1B-to-mcore
CHECKPOINT_PATH=${CURRENT_DIR}/checkpoints/qwen3vl-5b-a1b-hfm4
TENSORBOARD_LOGS_PATH=${CURRENT_DIR}/tensorboard_logs/qwen3vl-5b-a1b-hfm4
TRAIN_ITERS=5000000
LR_WARMUP_ITERS=50
LR_DECAY_ITERS=450

MODEL_ARGS=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 32
    --hidden-size 1024
    --ffn-hidden-size 3072
    --moe-ffn-hidden-size 640
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --num-attention-heads 16
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --max-padding-length ${SEQ_LEN}
    --position-embedding-type rope
    --group-query-attention
    --num-query-groups 4
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function sigmoid
    --moe-router-topk 4
    --moe-layer-freq "([1]*32)"
    --num-experts 64
    --patch-size 16
    --qk-layernorm
    --kv-channels 64
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --rotary-base 100000
    --rotary-seq-len-interpolation-factor 1
    --rotary-percent 1.0
    --padded-vocab-size 282742
    --patch-tokenizer-type Qwen2VLTokenizer
    --freeze-ViT
)

TRAINING_ARGS=(
    --use-mcore-models
    --load ${PRETRAIN_CHECKPOINT_PATH}
    --save "$CHECKPOINT_PATH"
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 2.0e-4
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --train-data-path ${TRAIN_DATA_PATH}
    --valid-data-path ${VALID_DATA_PATH}
    --split 99,1,0
    --num-workers 0
    --disable-vision-class-token
    --dataloader-type external
    --distributed-timeout-minutes 60
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --save-interval 1000
    --eval-iters 32
    --eval-interval 1000
    --dist-ckpt-strictness log_all
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

INFRA_ARGS=(
    --enable-experimental
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${ETP}
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend flash
    --recompute-granularity selective
    --overlap-grad-reduce
    --overlap-param-gather
)

exec hyperpodrun \
    --nnodes=${NNODES} --nproc-per-node=${NPROC_PER_NODE} \
    --server-host=0.0.0.0 --server-port=8080 \
    --tee=3 --log_dir=${CURRENT_DIR}/hyperpodrun-logs --server-log-level=warning \
    pretrain_qwen.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]} \
    2>&1 | tee run_qwen3_vl.log