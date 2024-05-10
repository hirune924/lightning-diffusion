from lightning import LightningDataModule
from pathlib import Path
import os
import torch
import random
import numpy as np
from PIL import Image
import datasets as hfd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import Any

class HFDataset(Dataset):
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

class HFStableDiffusionDataset(HFDataset):
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

class HFDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        image_column: str = "image",
        caption_column: str = "text",
        csv: str = "metadata.csv",
        cache_dir: str | None = None,
        dataset_cls: type[HFDataset] = HFStableDiffusionDataset,
        batch_size: int= 8,
        num_workers: int=4,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.caption_column = caption_column
        self.csv = csv
        self.cache_dir = cache_dir
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.num_workers = num_workers

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        if not Path(self.dataset_name).exists():
            hfd.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = self.dataset_cls(
                self.dataset_name, self.image_column, self.caption_column,
                self.csv, self.cache_dir,
                )
            self.dataset.init_post_process()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)
