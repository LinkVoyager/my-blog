#!/usr/bin/env bash
# 整理时新增：保持原训练参数，支持从本目录启动全部五组。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset WANDB_RUN_ID WANDB_RESUME
python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-model/Qwen2.5-Math-1.5B}"
train_data="${TRAIN_DATA_PATH:-data/gsm8k/train_sft_reason_gsm8k_raw.jsonl}"
val_data="${VAL_DATA_PATH:-data/gsm8k/test.jsonl}"
experiment_root="${OUTPUT_ROOT:-result/five_sizes}"
[[ -d "$model_path" ]] || { echo "模型目录不存在：$model_path" >&2; exit 1; }
[[ -f "$train_data" && -f "$val_data" ]] || { echo "请设置训练和评估 JSONL 路径。" >&2; exit 1; }
mkdir -p "$experiment_root"
for size in 128 256 512 1024 7473; do
    run_name="sft_size${size}_steps256_gsm8k_single_gpu"
    run_dir="${experiment_root}/${run_name}"
    mkdir "$run_dir"
    echo "开始训练：${run_name}"
    "$python_bin" -u -m learning.train_sft_step \
        --model_id "$model_path" \
        --train_data_path "$train_data" \
        --val_data_path "$val_data" \
        --prompt_path cs336_alignment/prompts/r1_zero.prompt \
        --output_dir "${run_dir}/checkpoints" \
        --dataset_size "$size" --max_steps 256 --lr 2e-5 \
        --batch_size 64 --micro_batch_size 2 --gradient_accumulation_steps 32 \
        --seed 42 --max_tokens 1024 --device cuda:0 --vllm_device cuda:0 \
        --vllm_gpu_util 0.3 --eval_every_steps 8 --max_eval_samples 1319 \
        --wandb_project sft_learning_blog --wandb_run_name "$run_name" \
        > "${run_dir}/train.log" 2>&1
    echo "完成训练：${run_name}"
done
