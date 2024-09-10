from pathlib import Path
import os
import torch
import random
import numpy as np
from PIL import Image
import datasets as hfd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Any
from lightning_diffusion.data.transforms import RandomCropWithInfo, ComputeTimeIds, T5TextPreprocess
from transformers import AutoProcessor
from diffusers.utils import load_image

class HFGeneralDataset(Dataset):
    """Dataset for huggingface datasets.

    Args:
    ----
        dataset: Dataset name or path to dataset.
        image_column: Image column name. Defaults to 'image'.
        caption_column: Caption column name. Defaults to 'text'.
        condition_column: Condition column name for ControlNet. Defaults to 'condition'.
        csv: Caption csv file name when loading local folder. Defaults to 'metadata.csv'.
        cache_dir: The directory where the downloaded datasets will be stored.Defaults to None.
    """
    def __init__(self,
                 dataset: str,
                 column_map: dict[str, str] = {"image": "image", "condition_img": "condition", "text": "text"},
                 csv: str = "metadata.csv",
                 cache_dir: str | None = None) -> None:
        self.dataset_name = dataset
        if Path(dataset).exists():
            # load local folder
            data_file = os.path.join(dataset, csv)
            self.dataset = hfd.load_dataset(
                "csv", data_files=data_file, cache_dir=cache_dir)["train"]
        else:
            # load huggingface online
            self.dataset = hfd.load_dataset(dataset, cache_dir=cache_dir)["train"]

        self.column_map = column_map

    def __len__(self) -> int:
        """Get the length of dataset.

        Returns
        -------
            int: The length of filtered dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Args:
        ----
            idx: The index of self.data_list.

        Returns:
        -------
            dict: The idx-th image and data information of dataset after `post_process()`.
        """
        data_info = self.dataset[idx]
        result = {}
        for k, v in self.column_map.items():
            result[k] = data_info[v]
            
        return self.post_process(result)
    
    def init_post_process(self):
        raise NotImplementedError()
    
    def post_process(self):
        raise NotImplementedError()
    
class HFStableDiffusionControlnetDataset(HFGeneralDataset):
    def init_post_process(self):
        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=512, interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop(size=512),
            v2.RandomHorizontalFlip(),
            #v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])
    def post_process(self, input: dict[str: Any]):
        if isinstance(input["image"], str):
            input["image"] = load_image(os.path.join(self.dataset_name, input["image"]))
        input["image"].convert('RGB')
        if isinstance(input["condition_img"], str):
            input["condition_img"] = load_image(os.path.join(self.dataset_name, input["condition_img"]))
        input["condition_img"].convert('RGB')
        if isinstance(input["text"], list | np.ndarray):
            input["text"] = random.choice(input["text"])

        input['image'], input['condition_img'] = self.transform(input['image'], input['condition_img'])
        return input
    
class HFStableDiffusionXLControlnetDataset(HFGeneralDataset):
    def init_post_process(self):
        self.resize = v2.Resize(size=1024, interpolation=v2.InterpolationMode.BILINEAR)
        self.hflip = v2.RandomHorizontalFlip(p=0.5)
        self.random_crop = RandomCropWithInfo(size=1024)
        self.time_ids = ComputeTimeIds()
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])

    def post_process(self, input: dict[str: Any]):
        if isinstance(input["image"], str):
            input["image"] = load_image(os.path.join(self.dataset_name, input["image"]))
        input["image"].convert('RGB')
        if isinstance(input["condition_img"], str):
            input["condition_img"] = load_image(os.path.join(self.dataset_name, input["condition_img"]))
        input["condition_img"].convert('RGB')
        if isinstance(input["text"], list | np.ndarray):
            input["text"] = random.choice(input["text"])

        original_img_shape =  [input['image'].height, input['image'].width]
        input['image'], input['condition_img'] = self.resize(input['image'], input['condition_img'])
        input['image'], input['condition_img'] = self.hflip(input['image'], input['condition_img'])
        (input['image'], input['condition_img']), size_info = self.random_crop([input['image'], input['condition_img']])
        size_info['original_img_shape'] = original_img_shape
        input["time_ids"] = self.time_ids(input['image'], size_info)
        input['image'], input['condition_img'] = self.normalize(input['image'], input['condition_img'])
        
        return input
    
class HFFluxControlnetDataset(HFGeneralDataset):
    def init_post_process(self, image_size: int = 512):
        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=image_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop(size=image_size),
            v2.RandomHorizontalFlip(),
            #v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])

    def post_process(self, input: dict[str: Any]):
        if isinstance(input["image"], str):
            input["image"] = load_image(os.path.join(self.dataset_name, input["image"]))
        input["image"].convert('RGB')
        if isinstance(input["condition_img"], str):
            input["condition_img"] = load_image(os.path.join(self.dataset_name, input["condition_img"]))
        input["condition_img"].convert('RGB')
        if isinstance(input["text"], list | np.ndarray):
            input["text"] = random.choice(input["text"])

        input['image'], input['condition_img'] = self.transform(input['image'], input['condition_img'])
        
        return input