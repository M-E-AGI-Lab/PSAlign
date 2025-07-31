"""
Sage Dataset for Personalized Safety Alignment (PSA) DPO Training.

Supports:
1. Paired image preferences
2. Caption tokenization
3. User embedding integration
4. SD / SDXL model compatibility
"""

import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class SageDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer,
        embeds_folder: str = None,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
        proportion_empty_prompts: float = 0.0,
        sdxl: bool = False
    ):
        if not sdxl:
            assert tokenizer is not None, "Tokenizer must be provided for non-SDXL mode"

        with open(metadata_path, 'r', encoding='utf-8') as f: 
            self.records = [json.loads(line) for line in f]

        self.image_root = image_root
        self.tokenizer = tokenizer if not sdxl else None
        self.embeds_folder = embeds_folder
        self.proportion_empty = proportion_empty_prompts
        self.sdxl = sdxl

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution) if random_crop else transforms.CenterCrop(resolution),
            transforms.Lambda(lambda x: x) if no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.records)

    def _load_pil(self, rel_path: str) -> Image.Image:
        img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        return img

    def _tokenize(self, caption: str):
        if random.random() < self.proportion_empty:
            caption = ""
        if self.sdxl:
            return caption
        return self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)


    def _load_user_emb(self, user_data: dict) -> torch.Tensor:
        uid = user_data["User_ID"]
        path = os.path.join(self.embeds_folder, f"{uid}.npy")
        if os.path.exists(path):
            emb = np.load(path).squeeze()
        else:
            emb = np.zeros(3584, dtype=np.float32)
            print(f"Embedding for user {uid} not found, using zeros.")
        return torch.tensor(emb, dtype=torch.float16)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        try:
            im0 = self.transform(self._load_pil(rec['chosen']))
            im1 = self.transform(self._load_pil(rec['rejected']))
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        out = {'pixel_values': torch.cat([im0, im1], dim=0)}
        caption = rec.get('caption', "")
        token_or_text = self._tokenize(caption)
        out['caption' if self.sdxl else 'input_ids'] = token_or_text

        if self.embeds_folder and 'user_data' in rec:
            out['user_embedding'] = self._load_user_emb(rec['user_data'])

        return out


def collate_fn(batch):
    out = {'pixel_values': torch.stack([ex['pixel_values'] for ex in batch]).to(memory_format=torch.contiguous_format).float()}
    if 'input_ids' in batch[0]:
        out['input_ids'] = torch.stack([ex['input_ids'] for ex in batch])
    if 'caption' in batch[0]:
        out['caption'] = [ex['caption'] for ex in batch]
    if 'user_embedding' in batch[0]:
        out['user_embedding'] = torch.stack([ex['user_embedding'] for ex in batch]).float()
    return out


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Subset

    config = dict(
        sdxl=False,
        sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        revision=None,
        metadata="../data/sage/train/metadata.jsonl",
        image_root="../data/sage/train",
        embeds_folder="../data/embeddings",
        resolution=512,
        random_crop=False,
        no_hflip=False,
        proportion_empty_prompts=0.2,
        batch_size=24,
        num_workers=32,
        max_train_samples=1000,
        seed=2025
    )

    tokenizer = AutoTokenizer.from_pretrained(config["sd_model"], subfolder="tokenizer", revision=config["revision"])

    ds = SageDataset(
        metadata_path=config["metadata"],
        image_root=config["image_root"],
        tokenizer=tokenizer,
        embeds_folder=config["embeds_folder"],
        resolution=config["resolution"],
        random_crop=config["random_crop"],
        no_hflip=config["no_hflip"],
        proportion_empty_prompts=config["proportion_empty_prompts"],
        sdxl=config["sdxl"]
    )

    if config["max_train_samples"] is not None:
        total = len(ds)
        g = torch.Generator().manual_seed(config["seed"])
        indices = torch.randperm(total, generator=g)[:config["max_train_samples"]]
        ds = Subset(ds, indices)

    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        drop_last=True
    )

    for step, batch in enumerate(tqdm(loader)):
        print({k: (v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()})
