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
import json

class AnimateAnyonePose2ImgDataset(Dataset):
    def __init__(
        self,
        data_path,
        img_size=512,
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # 　　{'video_path': , 'kps_path': , 'other':}
        # 　　{'video_path': , 'kps_path': , 'other':}
        # -----
        if isinstance(data_path, str):
            data_path = [data_path]
        vid_meta = []
        for data_meta_path in data_path:
            with open(data_meta_path, "r") as f:
                for line in f:
                    vid_meta.append(json.loads(line))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
    def __len__(self):
        return len(self.vid_meta)
    
    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

class AnimateAnyonePose2VidDataset(Dataset):
    def __init__(
        self,
        data_path,
        sample_rate=4,
        n_sample_frames=24,
        width=512,
        height=512
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.sample_rate = sample_rate  
        self.n_sample_frames = n_sample_frames
        # -----
        # vid_meta format:
        # 　　{'video_path': , 'kps_path': , 'other':}
        # 　　{'video_path': , 'kps_path': , 'other':}
        # -----
        if isinstance(data_path, str):
            data_path = [data_path]
        vid_meta = []
        for data_meta_path in data_path:
            with open(data_meta_path, "r") as f:
                for line in f:
                    vid_meta.append(json.loads(line))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.vid_meta)
    
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

class AnimateAnyoneSketch2ColorDataset(Dataset):
    def __init__(
        self,
        data_path,
        img_size=512,
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # 　　{'video_path': , 'kps_path': , 'other':}
        # 　　{'video_path': , 'kps_path': , 'other':}
        # -----
        if isinstance(data_path, str):
            data_path = [data_path]
        vid_meta = []
        for data_meta_path in data_path:
            with open(data_meta_path, "r") as f:
                for line in f:
                    vid_meta.append(json.loads(line))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(1.0, 1.0),
                    ratio=(0.9, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
    def __len__(self):
        return len(self.vid_meta)
    
    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]

        video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        import cv2
        from PIL import ImageEnhance
        from lightning_diffusion.data.animate_anyone.util.extract_line import XDoG_filter
        tgt_sketch_pil = tgt_img_pil.copy()
        mode = random.choice(['xdog', 'gray', 'gray+xdog', 'gray+thresh'])
        if mode == 'xdog':
            condition_image = np.array(tgt_sketch_pil)
            condition_image = XDoG_filter(condition_image, 
                    kernel_size=0,
                    sigma=1.4,
                    k_sigma=1.6,
                    epsilon=0,
                    phi=10,
                    gamma=0.98)
            condition_image = Image.fromarray(condition_image).convert("L").convert("RGB")
        elif mode == 'gray':
            condition_image = tgt_sketch_pil.convert("L").convert("RGB")
        elif mode == 'gray+xdog':
            condition_image = np.array(tgt_sketch_pil)
            xdog_edge = XDoG_filter(condition_image, 
                    kernel_size=0,
                    sigma=1.4,
                    k_sigma=1.6,
                    epsilon=0,
                    phi=10,
                    gamma=0.98)
            condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2GRAY)
            condition_image = cv2.cvtColor(condition_image, cv2.COLOR_GRAY2RGB)
            xdog_edge = cv2.cvtColor(xdog_edge, cv2.COLOR_RGB2GRAY)
            xdog_edge = cv2.cvtColor(xdog_edge, cv2.COLOR_GRAY2RGB)
            condition_image = cv2.addWeighted(condition_image, 0.5, xdog_edge, 0.5, 0)
            condition_image = Image.fromarray(condition_image).convert("L").convert("RGB")
        elif mode == 'gray+thresh':
            condition_image = np.array(tgt_sketch_pil.convert("L"))
            threshold = random.randint(100, 150)
            _, condition_image = cv2.threshold(condition_image, threshold, 255, cv2.THRESH_BINARY)
            condition_image = Image.fromarray(condition_image).convert("RGB")
        # Randomly adjust brightness of condition_image
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.5, 1.5)  # Adjust range as needed
            enhancer = ImageEnhance.Brightness(condition_image)
            condition_image = enhancer.enhance(brightness_factor)


        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(condition_image, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

class AnimateAnyonePose2ImgDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str|list[str],
        batch_size: int= 8,
        num_workers: int=4,
        dataset_args: dict[str, Any] = {"img_size": 512, "sample_margin": 30},
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_args = dataset_args

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = AnimateAnyonePose2ImgDataset(data_path=self.data_path,
                                                **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)

class AnimateAnyonePose2VidDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str|list[str],
        batch_size: int= 1,
        num_workers: int=4,
        dataset_args: dict[str, Any] = {"sample_rate": 4, "n_sample_frames": 24, "width": 512, "height": 512},
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_args = dataset_args

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = AnimateAnyonePose2VidDataset(data_path=self.data_path,
                                                **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)
    

class AnimateAnyoneSketch2ColorDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str|list[str],
        batch_size: int= 8,
        num_workers: int=4,
        dataset_args: dict[str, Any] = {"img_size": 512, "sample_margin": 30},
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_args = dataset_args

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = AnimateAnyoneSketch2ColorDataset(data_path=self.data_path,
                                                **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)