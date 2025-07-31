import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


class ImageGenerator:
    """Multi-GPU image generation using Stable Diffusion with Accelerate"""
    
    def __init__(
        self,
        accelerator: Accelerator,
        seed: int,
        model_path: str,
        lora_path: str = None,
        lora_scale: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 40,
    ):
        """
        Initialize the image generator
        
        Args:
            accelerator: Accelerate instance for multi-GPU support
            seed: Random seed for reproducibility
            model_path: Path to Stable Diffusion model
            lora_path: Path to LoRA weights (optional)
            lora_scale: LoRA scaling factor
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
        """
        # Store accelerator instance
        self.accelerator = accelerator

        # Load appropriate Diffusion Pipeline
        if 'xl' in model_path.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )

        # Load LoRA weights if provided
        if lora_path:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_scale)

        # Disable safety checker and progress bar
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)

        # Prepare model for multi-GPU distribution
        self.pipe = accelerator.prepare(pipe)
        self.pipe.to(accelerator.device)

        # Store generation parameters
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator = torch.Generator(accelerator.device).manual_seed(seed)

    def generate_images(self, input_file: str, output_folder: str, batch_size: int = 1) -> bool:
        """
        Generate images from CSV prompts file
        
        Args:
            input_file: Path to CSV file containing prompts
            output_folder: Directory to save generated images
            batch_size: Number of images to generate per batch
            
        Returns:
            bool: True if generation completed successfully
        """
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Validate input file format
        if not input_file.endswith('.csv'):
            raise ValueError('Invalid input file format. Expected CSV file.')

        # Load prompts from CSV
        df = pd.read_csv(input_file, lineterminator='\n')
        records = df.to_dict(orient='records')

        # Distribute work across processes
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        subset = records[rank::world_size]

        # Process batches
        for idx in tqdm(
            range(0, len(subset), batch_size),
            desc=f"Rank {rank}",
            disable=not self.accelerator.is_main_process
        ):
            batch = subset[idx: idx + batch_size]
            
            # Handle different caption column names
            prompt_column = 'caption' if 'caption' in batch[0] else 'caption\r'
            prompts = [record[prompt_column] for record in batch]

            # Set up generator (support evaluation_seed override)
            generator = self.generator
            if 'evaluation_seed' in batch[0]:
                generator = torch.Generator(self.accelerator.device).manual_seed(
                    batch[0]['evaluation_seed']
                )

            # Generate images
            images = self.pipe(
                prompts,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            ).images

            # Save generated images
            for record, image in zip(batch, images):
                filename = record['file_name']
                image.save(os.path.join(output_folder, filename))

        # Wait for all processes to complete
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            print("Image generation completed successfully.")
        
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Stable Diffusion Inference with Accelerate",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to Stable Diffusion model'
    )
    parser.add_argument(
        '--lora_path', 
        type=str, 
        default=None,
        help='Path to LoRA weights (optional)'
    )
    parser.add_argument(
        '--lora_scale', 
        type=float, 
        default=1.0,
        help='LoRA scaling factor (default: 1.0)'
    )
    parser.add_argument(
        '--prompts_path', 
        type=str, 
        required=True,
        help='Path to CSV file containing prompts'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True,
        help='Directory to save generated images'
    )
    parser.add_argument(
        '--guidance_scale', 
        type=float, 
        default=7.5,
        help='Classifier-free guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=40,
        help='Number of denoising steps (default: 40)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=2025,
        help='Random seed for reproducibility (default: 2025)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size for generation (default: 1)'
    )
    
    return parser.parse_args()


def main():
    """Main function for multi-GPU image generation"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()
    
    # Create image generator instance
    generator = ImageGenerator(
        accelerator=accelerator,
        seed=args.seed,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )
    
    # Generate images from prompts
    generator.generate_images(
        input_file=args.prompts_path,
        output_folder=args.save_path,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()