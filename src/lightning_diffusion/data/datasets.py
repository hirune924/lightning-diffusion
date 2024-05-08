from lightning import LightningDataModule
from pathlib import Path
import os
import torch
#from datasets import load_dataset, Dataset, DatasetDict
import datasets as hfd
from torch.utils.data import Dataset, DataLoader

class HFStableDiffusionDataset(Dataset):
    def __init__(self, dataset: hfd.Dataset | hfd.DatasetDict):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        item = self.ds[index]
        return item


class HFDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        image_column: str = "image",
        caption_column: str = "text",
        csv: str = "metadata.csv",
        dataset_cls: type[HFStableDiffusionDataset] = HFStableDiffusionDataset,
        batch_size: int= 8,
        cache_dir: str | None = None
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.caption_column = caption_column
        self.csv = csv
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    def prepare_data(self):
        if not Path(self.dataset_name).exists():
            hfd.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]

    def setup(self, stage: str):
        if Path(self.dataset_name).exists():
            # load local folder
            data_file = os.path.join(self.dataset_name, self.csv)
            ds = hfd.load_dataset(
                "csv", data_files=data_file, cache_dir=self.cache_dir)["train"]
        else:
            # load huggingface online
            ds = hfd.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]

        self.dataset = self.dataset_cls(ds)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
