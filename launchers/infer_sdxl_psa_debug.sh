CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch infer.py \
    --sdxl --sd_model stabilityai/stable-diffusion-xl-base-1.0 \
    --meta_data ./data/sage/val/metadata.jsonl \
    --save_path ./infer_results/sdxl-debug --batch_size 1 --psa_scale 1.0 \
    --load_psa --psa_path ./trained_models/psa-sdxl-debug \
    --llm_model Qwen/Qwen2.5-7B-Instruct --embeds_folder ./data/embeddings