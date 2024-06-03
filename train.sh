#!/bin/bash
uname -a
date
DATA_PATH='/home/tsl/data/HumanParsing/shhq_cihp/train_data'
# 打印DATA_PATH数据路径
echo "data path: ${DATA_PATH}"

NUM_CLASSES=20
INPUT_SIZE='512,512'
BATCH_SIZE=8
GPU_NUMS='0'
EPOCHS=200
EVAL_EPOCHS=2
LOG_AND_SAVE_PATH='./log_shhq_cihp'

# 获取new_checkpoint.txt中的内容并拼接
NEW_CHECKPOINT_FILE="${LOG_AND_SAVE_PATH}/new_checkpoint.txt"
if [ -f "${NEW_CHECKPOINT_FILE}" ]; then
    NEW_CHECKPOINT=$(cat ${NEW_CHECKPOINT_FILE})
    RESTORE_FROM="${LOG_AND_SAVE_PATH}/${NEW_CHECKPOINT}"
else
    RESTORE_FROM="${LOG_AND_SAVE_PATH}/checkpoint_160.pth.tar"
fi

echo "restore from: ${RESTORE_FROM}"

# 设置最大内存使用量为 50GB
# ulimit -v 52428800

# 限制程序可见GPU为GPU_NUMS中的值，并增加提示输出
export CUDA_VISIBLE_DEVICES=${GPU_NUMS}
echo "Using GPU: ${GPU_NUMS}"

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
