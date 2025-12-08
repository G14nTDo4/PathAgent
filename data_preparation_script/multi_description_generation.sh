#!/bin/bash
echo "Launching 4 parallel tasks..."

PROJECT_ROOT="Quilt_LLaVA_DIRECTORY" # change this to your project root directory
DATASET_ROOT="RESULTS_DIRECTORY" # change this to your dataset root directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

GPUS=(0 1 2 3) # change this to your GPU IDs for each task


for i in {1..4}
do
    GPU_ID=${GPUS[$((i-1))]}
    echo "Starting Task $i on GPU $GPU_ID..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/description_generation.py" \
        --model-path "$PROJECT_ROOT/qllava_ckpt/" \
        --image-dir "$DATASET_ROOT/patches_output" \
        --output-json "$DATASET_ROOT/desc/patches_descriptions${i}.json" \
        --slide-list "$DATASET_ROOT/split_name/slides_part${i}.txt" \
        --load-4bit > "$LOG_DIR/task${i}.log" 2>&1 &
done

echo "✅ All tasks started successfully. Please check $LOG_DIR for logs."
