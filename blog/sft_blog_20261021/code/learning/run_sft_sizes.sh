#!/usr/bin/env bash
set -euo pipefail

cd /home/lab/zhengkw/work/cs336-5

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_ENTITY=wangsiyuan

# 每组创建独立的实验记录。
unset WANDB_RUN_ID WANDB_RESUME

experiment_root="result/sft_size_comparison_20261021"
mkdir -p "$experiment_root"

for size in 128 256 512 1024; do
    run_name="sft_size${size}_steps256_gsm8k_single_gpu"
    run_dir="${experiment_root}/${run_name}"

    # 目录已存在时停止，避免覆盖以前的结果。
    mkdir "$run_dir"

    echo "开始训练：${run_name}"

    /home/lab/zhengkw/anaconda3/envs/cs336/bin/python \
        -u -m learning.train_sft_step \
        --model_id model/Qwen2.5-Math-1.5B \
        --train_data_path data/gsm8k/train_sft_reason_gsm8k_raw.jsonl \
        --val_data_path data/gsm8k/test.jsonl \
        --prompt_path cs336_alignment/prompts/r1_zero.prompt \
        --output_dir "${run_dir}/checkpoints" \
        --dataset_size "$size" \
        --max_steps 256 \
        --lr 2e-5 \
        --batch_size 64 \
        --micro_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --seed 42 \
        --max_tokens 1024 \
        --device cuda:0 \
        --vllm_device cuda:0 \
        --vllm_gpu_util 0.3 \
        --eval_every_steps 8 \
        --max_eval_samples 1319 \
        --wandb_project cs335_5_sft \
        --wandb_run_name "$run_name" \
        > "${run_dir}/train.log" 2>&1

    echo "完成训练：${run_name}"
done

echo "四组实验全部完成。"