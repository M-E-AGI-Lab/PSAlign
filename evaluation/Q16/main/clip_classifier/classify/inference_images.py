import fire
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import fsspec
from main.paper_experiments.experiments import run_model_imagefolder
from argparse import Namespace


clip_model_name = 'ViT-L/14'
prompt_path = f'data/{clip_model_name.replace("/", "-")}/prompts.p'


def main_imagedataset(input_folder, output_folder):
    """main function"""
    args = Namespace(
        language_model='Clip_'+clip_model_name,
        model_type='sim',
        prompt_path=prompt_path,
        only_inappropriate=False,
        input_type='img',
        gpu=[0],
    )
    run_model_imagefolder(args, input_folder, output_folder)


if __name__ == '__main__':
    fire.Fire(main_imagedataset)

# cd ~/workspace/Q16
# CUDA_VISIBLE_DEVICES=3 python main/clip_classifier/classify/inference_images.py \
# --input_folder ~/workspace/SafetyDPO/CoProv2 \
# --output_folder results_CoProv2