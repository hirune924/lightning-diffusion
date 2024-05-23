from lightning import LightningDataModule
from pathlib import Path
import datasets as hfd
from torch.utils.data import DataLoader
from typing import Any, Union
from lightning_diffusion.data.t2i_datasets import HFT2IDataset, HFStableDiffusionDataset
from lightning_diffusion.data.controlnet_datasets import HFControlnetDataset
class HFDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        cache_dir: str | None = None,
        batch_size: int= 8,
        num_workers: int=4,
        dataset_cls: type[HFT2IDataset] | type[HFControlnetDataset] = HFStableDiffusionDataset,
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
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, persistent_workers=True)
