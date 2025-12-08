#!/bin/bash

PROJECT_ROOT="PLIP_DIRECTORY" # change this to your PLIP project root directory
DATASET_ROOT="RESULTS_DIRECTORY" # change this to your dataset root directory
PLIP_CHECKPOINT_PATH="PLIP_CHECKPOINT_PATH" # change this to your PLIP checkpoint path

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

echo "Starting task processes..."

GPUS=(0 1 2 3) # change this to your GPU IDs for each task

for i in {1..4}
do
    TASK_FILE=$DATASET_ROOT/slides_part${i}.txt
    LOG_FILE=$LOG_DIR/slides_part${i}.log

    GPU_ID=${GPUS[$((i-1))]}
    echo "Starting task $i on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python "$PROJECT_ROOT/img_emb_generation.py" \
        --plip_lib_path $PROJECT_ROOT \
        --task_file $TASK_FILE \
        --img_root $DATASET_ROOT/patches_output \
        --output_dir $DATASET_ROOT/img_features \
        --plip_ckpt $PLIP_CHECKPOINT_PATH > $LOG_FILE 2>&1 &
done

echo "All tasks started, please check log files in $LOG_DIR"
