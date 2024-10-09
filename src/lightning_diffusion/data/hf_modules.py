from lightning import LightningDataModule
from pathlib import Path
import datasets as hfd
from torch.utils.data import DataLoader, RandomSampler
from typing import Any, Union
from lightning_diffusion.data import HFImageTextDataset, HFStableDiffusionDataset
from lightning_diffusion.data import HFGeneralDataset
from lightning_diffusion.data import HFImageDataset
from lightning_diffusion.data.data_sampler import AspectRatioBatchSampler

class HFDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        cache_dir: str | None = None,
        batch_size: int= 8,
        num_workers: int=4,
        multi_aspect: bool = False,
        dataset_cls: type[HFImageTextDataset | HFGeneralDataset | HFImageDataset] = HFStableDiffusionDataset,
        dataset_args: dict[str, Any] = {"image_column": "image", "caption_column": "text", "csv": "metadata.csv"},
        dataset_process_args: dict[str, Any] = {}
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_args = dataset_args
        self.dataset_process_args = dataset_process_args
        self.multi_aspect = multi_aspect
    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        if not Path(self.dataset_name).exists():
            hfd.load_dataset(self.dataset_name, cache_dir=self.cache_dir)["train"]

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = self.dataset_cls(
                dataset=self.dataset_name, 
                cache_dir=self.cache_dir,
                **self.dataset_args
                )
            self.dataset.init_post_process(**self.dataset_process_args)

    def train_dataloader(self):
        if self.multi_aspect:
            batch_sampler = AspectRatioBatchSampler(
                sampler=RandomSampler(self.dataset),
                batch_size=self.batch_size,
                drop_last=True
            )
            return DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=self.num_workers,
                            shuffle=False, pin_memory=False, drop_last=False, persistent_workers=True)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)
