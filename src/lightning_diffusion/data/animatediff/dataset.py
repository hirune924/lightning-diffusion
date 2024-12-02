import os, random
import numpy as np
from decord import VideoReader
from typing import Any
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor
from lightning import LightningDataModule
from pathlib import Path
import json

class AnimateDiffText2VidDataset(Dataset):
    def __init__(
        self,
        csv_path: str | list[str],
        data_dir: str,
        sample_rate: int = 4,
        n_sample_frames: int = 16,
        width: int = 512,
        height: int = 512
    ):
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = Path(data_dir)
        self.width = width
        self.height = height
        self.sample_rate = sample_rate  
        self.n_sample_frames = n_sample_frames
        # -----
        # vid_meta format:
        # 　　{'video_path': , 'kps_path': , 'other':}
        # 　　{'video_path': , 'kps_path': , 'other':}
        # -----
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        vid_meta = []
        for data_meta_path in csv_path:
            vid_meta.append(pd.read_csv(Path(self.data_dir) / data_meta_path))

        self.vid_meta = pd.concat(vid_meta, ignore_index=True)

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(max(width, height)),
            transforms.CenterCrop((width, height)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return len(self.vid_meta)

    def __getitem__(self, index):
        video_path = Path(self.data_dir) / self.vid_meta.loc[index, "video_path"]
        prompt = self.vid_meta.loc[index, "caption"]

        video_reader = VideoReader(str(video_path))
        video_length = len(video_reader)
        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_rate + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        pixel_values = self.pixel_transforms(pixel_values)

        sample = dict(
            video_dir=str(video_path),
            pixel_values=pixel_values,
            text=prompt
        )

        return sample
    
class AnimateDiffText2VidDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str | list[str], 
        data_dir: str,
        width: int = 512, 
        height: int = 512,
        sample_rate: int = 4, 
        n_sample_frames: int = 24,
        batch_size: int= 1,
        num_workers: int=4,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.batch_size = batch_size
        self.num_workers = num_workers

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = AnimateDiffText2VidDataset(csv_path=self.csv_path,
                                                      data_dir=self.data_dir,
                                                      width=self.width,
                                                      height=self.height,
                                                      sample_rate=self.sample_rate,
                                                      n_sample_frames=self.n_sample_frames)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)