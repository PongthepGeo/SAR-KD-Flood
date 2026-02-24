import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

RAW_MIN = -22.225124
RAW_MAX = 0.0

TRAIN_BANDS = ['VV', 'VH', 'B2', 'B3', 'B4', 'NDVI', 'NDWI']
VAL_BANDS = ['VV', 'VH']


class MultiChannelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.bands = TRAIN_BANDS if split == 'train' else VAL_BANDS

        self.img_dir = self.root_dir / split / 'img'
        self.lab_dir = self.root_dir / split / 'lab'

        label_files = sorted([f.name for f in self.lab_dir.glob("*.npy")])
        candidates = [f.replace('.npy', '') for f in label_files]

        self.patch_ids = []
        for pid in candidates:
            if all((self.img_dir / f"{pid}_{b}.npy").exists() for b in self.bands):
                self.patch_ids.append(pid)

        if not self.patch_ids:
            raise ValueError(f"No valid image-label pairs in {self.img_dir}")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        band = np.random.choice(self.bands)

        image = np.load(self.img_dir / f"{patch_id}_{band}.npy").astype(np.float32)
        image = np.clip((image - RAW_MIN) / (RAW_MAX - RAW_MIN), 0, 1)
        image = np.expand_dims(image, axis=-1)

        mask = np.load(self.lab_dir / f"{patch_id}.npy").astype(np.float32)

        if self.transform:
            t = self.transform(image=image, mask=mask)
            image, mask = t['image'], (t['mask'] > 0.5).long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy((mask > 0.5).astype(np.uint8)).long()

        return image, mask
