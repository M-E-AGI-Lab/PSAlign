CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch infer.py \
    --sdxl --sd_model stabilityai/stable-diffusion-xl-base-1.0 \
    --meta_data ./data/sage/val/metadata.jsonl \
    --save_path ./infer_results/sdxl-psa-llm --batch_size 1 --psa_scale 1.0 \
    --load_psa --psa_path ./trained_models/psa-sdxl \
    --llm_model Qwen/Qwen2.5-7B-Instruct \
    --llm_ban_prompt_file ./data/user_data/num_1k/get_banned.txt \
    --llm_emb_prompt_file ./data/user_data/num_1k/get_embeds.txt