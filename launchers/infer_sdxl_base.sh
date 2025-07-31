CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch infer.py \
    --sdxl --sd_model stabilityai/stable-diffusion-xl-base-1.0 \
    --meta_data ./data/sage/val/metadata.jsonl \
    --save_path ./infer_results/sdxl-base --batch_size 1