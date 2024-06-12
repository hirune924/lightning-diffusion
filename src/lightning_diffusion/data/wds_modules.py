from lightning import LightningDataModule
import webdataset as wds
from lightning_diffusion.data import WDSDataset

class WDSDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_urls: str,
        batch_size: int= 8,
        num_workers: int=4,
        epoch_length: int=1000,
        dataset_cls: type[WDSDataset] = WDSDataset,
        target_keys: str="encoder_hidden_states.npy latents.npy"
    ):
        super().__init__()
        self.dataset_urls = dataset_urls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = dataset_cls
        self.epoch_length = epoch_length
        self.target_keys = target_keys

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = self.dataset_cls(self.dataset_urls,
                                            self.batch_size,
                                            self.target_keys).get_dataset()

    def train_dataloader(self):
        loader = wds.WebLoader(self.dataset, 
                               batch_size=None, 
                               shuffle=False, 
                               num_workers=self.num_workers)
        # Unbatch, shuffle between workers, then rebatch.
        loader = loader.unbatched().shuffle(1000).batched(self.batch_size)
        # Since we are using resampling, the dataset is infinite; set an artificial epoch size.
        loader = loader.with_epoch(self.epoch_length)
        return loader
