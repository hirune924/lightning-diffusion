from collections.abc import Generator
from torch.utils.data import BatchSampler, Sampler
import torch
class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False,
                 **kwargs) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self._aspect_ratio_buckets: dict = {}
        self.bucket_id_cache = {}

    def __iter__(self) -> Generator:
        for idx in self.sampler:
            if idx not in self.bucket_id_cache.keys():
                data_info = self.sampler.data_source[idx]
                bucket_id = data_info['bucket_id']
                self.bucket_id_cache[idx] = bucket_id
            else:
                bucket_id = self.bucket_id_cache[idx]
            # find the closest aspect ratio
            if bucket_id not in self._aspect_ratio_buckets.keys():
                self._aspect_ratio_buckets[bucket_id] = [idx]
            else:
                self._aspect_ratio_buckets[bucket_id].append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(self._aspect_ratio_buckets[bucket_id]) == self.batch_size:
                #yield self._aspect_ratio_buckets[bucket_id]
                #del self._aspect_ratio_buckets[bucket_id]
                yield self._aspect_ratio_buckets.pop(bucket_id)

        if not self.drop_last:
            for v in self._aspect_ratio_buckets.values():
                if len(v) > 0:
                    yield v
                    del v
        self._aspect_ratio_buckets = {}
