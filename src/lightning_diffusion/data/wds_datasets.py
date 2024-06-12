import webdataset as wds

class WDSDataset():
    def __init__(self, urls: str, batch_size: int, target_keys="encoder_hidden_states.npy latents.npy"):
        self.urls = urls
        self.batch_size = batch_size
        self.target_keys = target_keys

    def get_dataset(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=True, shardshuffle=True, nodesplitter=wds.split_by_node)
            .shuffle(1000)
            .decode("pil")
            .to_tuple(self.target_keys)
            .map(self.transform)
            .batched(self.batch_size, partial=False)
        )
        return dataset
    
    def transform(self, sample):
        return sample
