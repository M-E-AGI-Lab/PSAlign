#!/bin/bash
set -e

############################
# Configuration Parameters #
############################

# Support passing parameters through environment variables
DATASET="${DATASET:-debug}"      # If not set, the default is debug
GPUS="${GPUS:-0,1,2,3}"          # If not set, the default is 0,1,2,3
SEED=2025
PORT="29501"

# Models
SD15_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_MODEL="stabilityai/stable-diffusion-xl-base-1.0"

# Adapter paths
psa_adapter_SD15="../trained_models/psa-sd15"
psa_adapter_SDXL="../trained_models/psa-sdxl"
USER_EMBED_FOLDER="../data/embeddings"

# Output paths
PROMPTS="./eval_data/${DATASET}.csv"
OUTPUT_BASE="./eval_images/${DATASET}"

# LoRA / Safety edit paths
LOADER_SD15_LORA="~/workspace/SafetyDPO/safetydpo-models/sd15/pytorch_lora_weights.safetensors"
LOADER_SDXL_LORA="~/workspace/SafetyDPO/safetydpo-models/sdxl/pytorch_lora_weights.safetensors"
ESDU_SD15_PATH="~/workspace/erasing/esd-models/sd15/esdu.safetensors"
ESDU_SDXL_PATH="~/workspace/erasing/esd-models/sdxl/esdu.safetensors"
UCE_SD15_PATH="~/workspace/unified-concept-editing/uce_models/sd15/uce.safetensors"
UCE_SDXL_PATH="~/workspace/unified-concept-editing/uce_models/sdxl/uce.safetensors"
SLD_CONCEPT="NSFW, Hate, Harassment, Violence, Self-Harm, Sexuality, Shocking, Propaganda"

############################
# Baseline Generation      #
############################

echo "--- Running Baseline SD15 Generation ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_psa.py \
  --sd_model $SD15_MODEL \
  --input_csv $PROMPTS \
  --output_folder "$OUTPUT_BASE/base/sd15" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running Baseline SDXL Generation ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_psa.py \
  --sd_model $SDXL_MODEL \
  --input_csv $PROMPTS \
  --output_folder "$OUTPUT_BASE/base/sdxl" \
  --batch_size 1 \
  --sdxl \
  --seed $SEED

############################
# PSA Generation  #
############################

for i in {0..5}; do
  level="level_$i"

  echo "--- Running PSA-Adapter SD15 Generation: $level ---"
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_psa.py \
    --sd_model $SD15_MODEL \
    --psa_adapter_path $psa_adapter_SD15 \
    --user_embeds_folder $USER_EMBED_FOLDER \
    --input_csv $PROMPTS \
    --output_folder "$OUTPUT_BASE/psa/sd15" \
    --level $level \
    --batch_size 1 \
    --seed $SEED

  echo "--- Running PSA-Adapter SDXL Generation: $level ---"
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_psa.py \
    --sd_model $SDXL_MODEL \
    --psa_adapter_path $psa_adapter_SDXL \
    --user_embeds_folder $USER_EMBED_FOLDER \
    --input_csv $PROMPTS \
    --output_folder "$OUTPUT_BASE/psa/sdxl" \
    --level $level \
    --batch_size 1 \
    --sdxl \
    --seed $SEED
done

############################
# SafetyDPO Generation     #
############################

echo "--- Running SafetyDPO (SD15) Generation ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_safetydpo.py \
  --model_path $SD15_MODEL \
  --lora_path $LOADER_SD15_LORA \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/safetydpo/sd15" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running SafetyDPO (SDXL) Generation ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch generate_safetydpo.py \
  --model_path $SDXL_MODEL \
  --lora_path $LOADER_SDXL_LORA \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/safetydpo/sdxl" \
  --batch_size 1 \
  --seed $SEED

############################
# ESD-U / UCE / SLD Models #
############################

echo "--- Running ESD-U (SD15) ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port $PORT generate_ce.py \
  --method esdu \
  --base_model $SD15_MODEL \
  --edit_weights_path $ESDU_SD15_PATH \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/esdu/sd15" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running ESD-U (SDXL) ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port $PORT generate_ce.py \
  --method esdu \
  --base_model $SDXL_MODEL \
  --edit_weights_path $ESDU_SDXL_PATH \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/esdu/sdxl" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running UCE (SD15) ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port $PORT generate_ce.py \
  --method uce \
  --base_model $SD15_MODEL \
  --edit_weights_path $UCE_SD15_PATH \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/uce/sd15" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running UCE (SDXL) ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port $PORT generate_ce.py \
  --method uce \
  --base_model $SDXL_MODEL \
  --edit_weights_path $UCE_SDXL_PATH \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/uce/sdxl" \
  --batch_size 1 \
  --seed $SEED

echo "--- Running SLD (SD15 only) ---"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --main_process_port $PORT generate_ce.py \
  --method sld \
  --base_model $SD15_MODEL \
  --sld_concept "$SLD_CONCEPT" \
  --sld_type Medium \
  --prompts_path $PROMPTS \
  --save_path "$OUTPUT_BASE/sld/sd15" \
  --batch_size 1 \
  --seed $SEED

echo "==================== All tasks completed ===================="
