#!/usr/bin/env python3
import argparse
import os
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU evaluation task runner")
    parser.add_argument("--gpus", type=str, default="4,5,6,7", 
                        help="Comma-separated list of GPU IDs, e.g. '0,1,2,3'")
    parser.add_argument("--output", type=str, default="eval_results",
                        help="Name of the output result file")
    parser.add_argument("--eval_dir", type=str, default="eval_images",
                        help="Directory containing images for evaluation")
    return parser.parse_args()

def save_result(output_file, dataset, method, model, level, metrics, score=None, state="skipped"):
    """Save evaluation result to CSV file"""
    import fcntl
    timestamp = datetime.datetime.now().strftime("%m-%d %H:%M")
    with open(output_file, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow([dataset, method, model, level, metrics, score if score is not None else "NA", state, timestamp])
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def run_inpro(gpu_id, dataset, method, model, level, output_file, eval_dir):
    """Run InPro evaluation"""
    target_folder = f"{eval_dir}/{dataset}/{method}/{model}"
    if level != "none":
        target_folder = f"{target_folder}/{level}"
    
    if not os.path.exists(target_folder):
        print(f"Skipping {dataset}/{method}/{model}/{level} (folder not found)")
        save_result(output_file, dataset, method, model, level, "inpro", state="skipped")
        return

    output_folder = f"{dataset}_{method}_{model}"
    if level != "none":
        output_folder += f"_{level}"
    output_folder += f"_gpu{gpu_id}"

    print(f"GPU[{gpu_id}] Running InPro: {dataset}/{method}/{model}/{level}")

    q16_cmd = f"""
    cd ./Q16 && \
    CUDA_VISIBLE_DEVICES={gpu_id} python main/clip_classifier/classify/inference_images.py \
    --input_folder ../{target_folder} \
    --output_folder {output_folder}
    """

    try:
        subprocess.run(q16_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"GPU[{gpu_id}] Q16 classifier failed: {dataset}/{method}/{model}/{level}")
        save_result(output_file, dataset, method, model, level, "inpro", state="failed")
        return

    tmp_output = f"{output_file}.tmp.{gpu_id}.csv"
    inpro_cmd = [
        "python", "eval.py",
        "--metrics", "inpro",
        "--target_folder", target_folder,
        "--reference", f"./Q16/data/{output_folder}/inappropriate/Clip_ViT-L/sim_prompt_tuneddata/inappropriate_images.csv",
        "--device", f"cuda:{gpu_id}",
        "--output_file", tmp_output
    ]

    try:
        subprocess.run(inpro_cmd, check=True)

        with open(tmp_output, 'r') as f:
            reader = list(csv.DictReader(f))
            if reader:
                score = reader[-1].get('score', 'NA')
                try:
                    score = float(score)
                except:
                    score = 'NA'
                save_result(output_file, dataset, method, model, level, "inpro", score, "succeeded")
            else:
                save_result(output_file, dataset, method, model, level, "inpro", state="failed")
        os.remove(tmp_output)
    except Exception as e:
        print(f"GPU[{gpu_id}] InPro task failed: {dataset}/{method}/{model}/{level}, error: {e}")
        save_result(output_file, dataset, method, model, level, "inpro", state="failed: " + str(e))

def run_clip(gpu_id, dataset, method, model, level, output_file, eval_dir):
    """Run CLIP Score evaluation"""
    target_folder = f"{eval_dir}/{dataset}/{method}/{model}"
    if level != "none":
        target_folder = f"{target_folder}/{level}"

    if not os.path.exists(target_folder):
        print(f"Skipping {dataset}/{method}/{model}/{level} (folder not found)")
        save_result(output_file, dataset, method, model, level, "clip", state="skipped")
        return

    print(f"GPU[{gpu_id}] Running CLIP: {dataset}/{method}/{model}/{level}")

    tmp_output = f"{output_file}.tmp.{gpu_id}.csv"
    cmd = [
        "python", "eval.py",
        "--metrics", "clip",
        "--target_folder", target_folder,
        "--reference", f"eval_data/{dataset}.csv",
        "--device", f"cuda:{gpu_id}",
        "--output_file", tmp_output
    ]

    try:
        subprocess.run(cmd, check=True)

        with open(tmp_output, 'r') as f:
            reader = list(csv.DictReader(f))
            if reader:
                score = reader[-1].get('score', 'NA')
                try:
                    score = float(score)
                except:
                    score = 'NA'
                save_result(output_file, dataset, method, model, level, "clip", score, "succeeded")
            else:
                save_result(output_file, dataset, method, model, level, "clip", state="failed")
        os.remove(tmp_output)
    except Exception as e:
        print(f"GPU[{gpu_id}] CLIP task failed: {dataset}/{method}/{model}/{level}, error: {e}")
        save_result(output_file, dataset, method, model, level, "clip", state="failed: " + str(e))

def run_fid(gpu_id, dataset, method, model, level, output_file, eval_dir):
    """Run FID evaluation"""
    target_folder = f"{eval_dir}/{dataset}/{method}/{model}"
    if level != "none":
        target_folder = f"{target_folder}/{level}"

    if not os.path.exists(target_folder):
        print(f"Skipping {dataset}/{method}/{model}/{level} (folder not found)")
        save_result(output_file, dataset, method, model, level, "fid", state="skipped")
        return

    print(f"GPU[{gpu_id}] Running FID: {dataset}/{method}/{model}/{level}")

    tmp_output = f"{output_file}.tmp.{gpu_id}.csv"
    cmd = [
        "python", "eval.py",
        "--metrics", "fid",
        "--target_folder", target_folder,
        "--reference", "~/workspace/PSAAdapter/data/coco-2014val-10k",
        "--device", f"cuda:{gpu_id}",
        "--output_file", tmp_output
    ]

    try:
        subprocess.run(cmd, check=True)

        with open(tmp_output, 'r') as f:
            reader = list(csv.DictReader(f))
            if reader:
                score = reader[-1].get('score', 'NA')
                try:
                    score = float(score)
                except:
                    score = 'NA'
                save_result(output_file, dataset, method, model, level, "fid", score, "succeeded")
            else:
                save_result(output_file, dataset, method, model, level, "fid", state="failed")
        os.remove(tmp_output)
    except Exception as e:
        print(f"GPU[{gpu_id}] FID task failed: {dataset}/{method}/{model}/{level}, error: {e}")
        save_result(output_file, dataset, method, model, level, "fid", state="failed: " + str(e))

def gpu_task_runner(gpu_id, tasks, output_file, eval_dir):
    """Run all evaluation tasks assigned to a single GPU"""
    print(f"GPU {gpu_id} starts processing {len(tasks)} tasks")

    for task in tasks:
        metrics, dataset, method, model, level = task
        try:
            if metrics == "clip":
                run_clip(gpu_id, dataset, method, model, level, output_file, eval_dir)
            elif metrics == "fid":
                run_fid(gpu_id, dataset, method, model, level, output_file, eval_dir)
            elif metrics == "inpro":
                run_inpro(gpu_id, dataset, method, model, level, output_file, eval_dir)
        except Exception as e:
            print(f"GPU[{gpu_id}] task error: {metrics} {dataset}/{method}/{model}/{level}, error: {e}")
            save_result(output_file, dataset, method, model, level, metrics, state="failed")

    print(f"GPU {gpu_id} completed all tasks")

def main():
    args = parse_args()
    gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    output_file = args.output
    eval_dir = args.eval_dir

    # Ensure result file exists and has header
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "method", "base_model", "level", "metrics", "score", "state", "timestamp"])

    # Load finished tasks from file
    finished_tasks = set()
    if os.path.exists(output_file):
        try:
            df_done = pd.read_csv(output_file)
            df_done = df_done[df_done["state"] == "succeeded"]
            for _, row in df_done.iterrows():
                finished_tasks.add((row["metrics"], row["dataset"], row["method"], row["base_model"], row["level"]))
        except Exception as e:
            print(f"Failed to read completed tasks: {e}")

    # Helper to add tasks not yet completed
    def add_task(tasks, finished_tasks, metrics, dataset, method, model, level):
        task = (metrics, dataset, method, model, level)
        if task not in finished_tasks:
            tasks.append(task)

    # Build task list
    tasks = []

    datasets = ["coco_10k", "i2p_4073", "CoProv2_test", "sage_unseen", "ud_1434"]
    methods = ["base", "esdu", "safetydpo", "sld", "uce", "psa"]
    models = ["sd15", "sdxl"]
    levels = ["none", "level_0", "level_2", "level_1", "level_3", "level_4"]

    # Add CLIP and FID tasks for coco_10k
    for method in methods:
        for model in models:
            if method == "sld" and model == "sdxl":
                continue  # sld only supports sd15

            if method == "psa":
                for level in levels:
                    if level == "none": continue
                    add_task(tasks, finished_tasks, "clip", "coco_10k", method, model, level)
                    add_task(tasks, finished_tasks, "fid", "coco_10k", method, model, level)
            else:
                add_task(tasks, finished_tasks, "clip", "coco_10k", method, model, "none")
                add_task(tasks, finished_tasks, "fid", "coco_10k", method, model, "none")

    # Add InPro tasks for other datasets
    for dataset in ["i2p_4073", "CoProv2_test", "sage_unseen", "ud_1434"]:
        for method in methods:
            for model in models:
                if method == "sld" and model == "sdxl":
                    continue
                if method == "psa":
                    for level in levels:
                        if level == "none": continue
                        add_task(tasks, finished_tasks, "inpro", dataset, method, model, level)
                else:
                    add_task(tasks, finished_tasks, "inpro", dataset, method, model, "none")

    print(f"Total tasks: {len(tasks)}")

    # Distribute tasks evenly among GPUs
    tasks_per_gpu = {gpu_id: [] for gpu_id in gpus}
    for i, task in enumerate(tasks):
        gpu_id = gpus[i % len(gpus)]
        tasks_per_gpu[gpu_id].append(task)

    for gpu_id, gpu_tasks in tasks_per_gpu.items():
        print(f"GPU {gpu_id}: {len(gpu_tasks)} tasks")

    # Run tasks in parallel using a process pool
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for gpu_id, gpu_tasks in tasks_per_gpu.items():
            futures.append(executor.submit(gpu_task_runner, gpu_id, gpu_tasks, output_file, eval_dir))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Task execution error: {exc}")

    # Summarize results
    try:
        df = pd.read_csv(output_file)
        completed = len(df[df['state'] == "succeeded"])
        skipped = len(df[df['state'] == "skipped"])
        total = len(df)

        print("\n===== Evaluation Summary =====")
        print(f"Total: {total}")
        print(f"Completed: {completed}")
        print(f"Skipped: {skipped}")

        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            if len(dataset_df) > 0:
                print(f"\n{dataset}:")
                for metric in ["clip", "fid", "inpro"]:
                    metric_df = dataset_df[dataset_df['metrics'] == metric]
                    if len(metric_df) > 0:
                        completed_metric = len(metric_df[metric_df['state'] == "succeeded"])
                        total_metric = len(metric_df)
                        print(f"  - {metric}: {completed_metric}/{total_metric} completed")
    except Exception as e:
        print(f"Error during result summary: {e}")

    print("\nAll evaluation tasks completed!")

if __name__ == "__main__":
    main()
