#!/bin/bash
uname -a
date
DATA_PATH='/home/tsl/data/HumanParsing/shhq_cihp/train_data'
#打印DATA_PATH数据路径
echo "data path: ${DATA_PATH}"

NUM_CLASSES=20
INPUT_SIZE='512,512'
BATCH_SIZE=8
GPU_NUMS='0'
EPOCHS=150
EVAL_EPOCHS=2
LOG_AND_SAVE_PATH='./log_shhq_cihp'
RESTORE_FROM='./log_shhq_cihp/checkpoint_140.pth.tar'

echo "restore from: ${RESTORE_FROM}"

# 设置最大内存使用量为 50GB
#ulimit -v 52428800

 python train.py \
    --data-dir ${DATA_PATH} \
    --num-classes ${NUM_CLASSES} \
    --input-size ${INPUT_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --gpu ${GPU_NUMS} \
    --epochs ${EPOCHS} \
    --eval-epochs ${EVAL_EPOCHS} \
    --log-dir ${LOG_AND_SAVE_PATH} \
    --model-restore ${RESTORE_FROM}

