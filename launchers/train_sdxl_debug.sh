#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Training
job_name="psa-sdxl-debug"
sd_model="stabilityai/stable-diffusion-xl-base-1.0"
llm_model="Qwen/Qwen2.5-7B-Instruct"
train_data_dir="./data/sage/train"
embeds_folder="./data/embeddings"
psa_path="./trained_models/$job_name"
save_path="./infer_results/$job_name"

mkdir -p $psa_path && cp "$0" "$psa_path/$(basename "$0").$(date +%Y%m%d%H%M%S)"

accelerate launch train.py --mixed_precision bf16 \
    --sdxl --sd_model $sd_model --llm_model $llm_model --output_dir $psa_path \
    --train_data_dir $train_data_dir --embeds_folder $embeds_folder \
    --train_batch_size 1 --dataloader_num_workers 1 --gradient_accumulation_steps 1 \
    --max_train_steps 10 --checkpointing_steps 5 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 5 \
    --learning_rate 5e-6 --beta_dpo 5000