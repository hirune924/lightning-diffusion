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

class HFT2IDataset(Dataset):
    """Dataset for huggingface datasets.

    Args:
    ----
        dataset: Dataset name or path to dataset.
        image_column: Image column name. Defaults to 'image'.
        caption_column: Caption column name. Defaults to 'text'.
        csv: Caption csv file name when loading local folder. Defaults to 'metadata.csv'.
        cache_dir: The directory where the downloaded datasets will be stored.Defaults to None.
    """
    def __init__(self,
                 dataset: str,
                 image_column: str = "image",
                 caption_column: str = "text",
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

        self.image_column = image_column
        self.caption_column = caption_column

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
        image = data_info[self.image_column]
        if isinstance(image, str):
            image = Image.open(os.path.join(self.dataset_name, image))
        image = image.convert("RGB")
        caption = data_info[self.caption_column]
        if isinstance(caption, str):
            pass
        elif isinstance(caption, list | np.ndarray):
            # take a random caption if there are multiple
            caption = random.choice(caption)
        else:
            msg = (f"Caption column `{self.caption_column}` should "
                   "contain either strings or lists of strings.")
            raise ValueError(msg)
        result = {"image": image, "text": caption}
        return self.post_process(result)
    
    def init_post_process(self):
        raise NotImplementedError()
    
    def post_process(self):
        raise NotImplementedError()

class HFStableDiffusionDataset(HFT2IDataset):
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
        input['image'] = self.transform(input['image'])
        return input

class HFStableDiffusionXLDataset(HFT2IDataset):
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
        original_img_shape =  [input['image'].height, input['image'].width]
        input['image'] = self.resize(input['image'])
        input['image'] = self.hflip(input['image'])
        input['image'], size_info = self.random_crop(input['image'])
        size_info['original_img_shape'] = original_img_shape
        input["time_ids"] = self.time_ids(input['image'], size_info)
        input['image'] = self.normalize(input['image'])
        
        return input

class HFStableDiffusionIPAdapterDataset(HFT2IDataset):
    def init_post_process(self, image_encoder: str):
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
        self.image_encoder_process = AutoProcessor.from_pretrained(image_encoder)

    def post_process(self, input: dict[str: Any]):
        input['clip_img'] = self.image_encoder_process(images=input['image'], return_tensors="pt").pixel_values[0]
        input['image'] = self.transform(input['image'])
        
        return input
    
class HFStableDiffusionXLIPAdapterDataset(HFT2IDataset):
    def init_post_process(self, image_encoder: str):
        self.resize = v2.Resize(size=1024, interpolation=v2.InterpolationMode.BILINEAR)
        self.hflip = v2.RandomHorizontalFlip(p=0.5)
        self.random_crop = RandomCropWithInfo(size=1024)
        self.time_ids = ComputeTimeIds()
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.image_encoder_process = AutoProcessor.from_pretrained(image_encoder)

    def post_process(self, input: dict[str: Any]):
        input['clip_img'] = self.image_encoder_process(images=input['image'], return_tensors="pt").pixel_values[0]
        
        original_img_shape =  [input['image'].height, input['image'].width]
        input['image'] = self.resize(input['image'])
        input['image'] = self.hflip(input['image'])
        input['image'], size_info = self.random_crop(input['image'])
        size_info['original_img_shape'] = original_img_shape
        input["time_ids"] = self.time_ids(input['image'], size_info)
        input['image'] = self.normalize(input['image'])
        
        return input
    
class HFPixArtAlphaDataset(HFT2IDataset):
    def init_post_process(self):
        self.resize = v2.Resize(size=512, interpolation=v2.InterpolationMode.BILINEAR)
        self.hflip = v2.RandomHorizontalFlip(p=0.5)
        self.random_crop = RandomCropWithInfo(size=512)
        self.time_ids = ComputeTimeIds()
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.t5_preprocess = T5TextPreprocess(clean_caption=True)

    def post_process(self, input: dict[str: Any]):
        input['resolution'] =  [float(input['image'].height), float(input['image'].width)]
        input['image'] = self.resize(input['image'])
        input['image'] = self.hflip(input['image'])
        input['image'], size_info = self.random_crop(input['image'])
        input['aspect_ratio'] = input['image'].height / input['image'].width

        input['image'] = self.normalize(input['image'])
        input['text'] = self.t5_preprocess(input['text'])
        
        return input