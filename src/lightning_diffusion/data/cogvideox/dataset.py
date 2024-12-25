import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler, BatchSampler, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from collections.abc import Generator
from lightning import LightningDataModule

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        self.num_videos = len(self.video_paths)
        if self.num_videos != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip) if random_flip else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, video, _ = self._preprocess_video(self.video_paths[index])

        return {
            "prompt": self.id_token + self.prompts[index],
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
        )

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone()# if self.image_to_video else None

        return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

    def get_bucket_info(self, index):
        video_reader = decord.VideoReader(uri=self.video_paths[index].as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
        )
        height, width, _ = video_reader[0].shape
        nearest_height, nearest_width = self._find_nearest_resolution(height, width)
        return (nearest_frame_bucket, nearest_height, nearest_width), self.video_paths[index]
    

class AspectRatioVideoBatchSampler(BatchSampler):
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
                (f, h, w), vpath = self.sampler.data_source.get_bucket_info(idx)
                bucket_id = (f, h, w)
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


class CogVideoXDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        video_reshape_mode: str = "center",
        batch_size: int = 1,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token
        self.height_buckets = height_buckets
        self.width_buckets = width_buckets
        self.frame_buckets = frame_buckets
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.video_reshape_mode = video_reshape_mode
        self.batch_size = batch_size
        self.num_workers = num_workers

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = VideoDatasetWithResizeAndRectangleCrop(data_root=self.data_root,
                                                      dataset_file=self.dataset_file,
                                                      caption_column=self.caption_column,
                                                      video_column=self.video_column,
                                                      max_num_frames=self.max_num_frames,
                                                      id_token=self.id_token,
                                                      height_buckets=self.height_buckets,
                                                      width_buckets=self.width_buckets,
                                                      frame_buckets=self.frame_buckets,
                                                      load_tensors=self.load_tensors,
                                                      random_flip=self.random_flip,
                                                      image_to_video=self.image_to_video,
                                                      video_reshape_mode=self.video_reshape_mode)

    def train_dataloader(self):
        batch_sampler = AspectRatioVideoBatchSampler(
            sampler=RandomSampler(self.dataset),
            batch_size=self.batch_size,
            drop_last=True
        )
        return DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=self.num_workers,
                        shuffle=False, pin_memory=False, drop_last=False, persistent_workers=True)
    
import webdataset as wds
import io
class WDSVideoDataset(Dataset):
    def __init__(
        self,
        urls: str,
        metadata: str,
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        random_flip: Optional[float] = None,
        video_reshape_mode: str = "center",
    ) -> None:
        super().__init__()

        self.urls = urls
        self.metadata = pd.read_parquet(metadata)
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.random_flip = random_flip
        self.video_reshape_mode = video_reshape_mode
        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]
        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip) if random_flip else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0
    
    def get_dataset(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=True, shardshuffle=True, nodesplitter=wds.split_by_node)
            .shuffle(1000)
            .decode()
            .map(self.preprocess, handler=wds.ignore_and_continue)#warn_and_continue
            .to_tuple('prompt', 'image', 'video')
            .batched(1, partial=False)
        )
        return dataset
    
    def preprocess(self, sample):
        target_df = self.metadata[self.metadata['url_link'] == sample['json']['url']]
        video = decord.VideoReader(io.BytesIO(sample['mp4']))
        target_item = target_df.sample(n=1).iloc[0]
        clip_frames = (target_item['clips'] * video.get_avg_fps()).astype(int)

        video_num_frames = clip_frames[1] - clip_frames[0]
        if video_num_frames < self.max_num_frames:
            assert False, f"Video {sample['json']['url']} has less than {self.max_num_frames} frames"
        nearest_frame_bucket = min(
            self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
        )

        if clip_frames[0] > clip_frames[1]:
            frame_indices = [clip_frames[0]]
        elif video_num_frames < self.max_num_frames:
            frame_indices = list(range(clip_frames[0], clip_frames[1]))
        else:
            #frame_indices = list(range(clip_frames[0], clip_frames[1], video_num_frames // self.max_num_frames))
            frame_indices = list(range(clip_frames[0], clip_frames[0]+self.max_num_frames))


        frames = video.get_batch(frame_indices)
        frames = frames[:self.max_num_frames].float()
        selected_num_frames = frames.shape[0]

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        remainder = (3 + (selected_num_frames % 4)) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone()

        del video
        import gc
        gc.collect()

        return {
            "prompt": self.id_token + target_item['text_description'],
            "image": image,
            "video": frames,
            #"video_metadata": {
            #    "num_frames": frames.shape[0],
            #    "height": frames.shape[2],
            #    "width": frames.shape[3],
            #},
        }

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr
    
class WDSVideoDataModule(LightningDataModule):
    def __init__(
        self,
        urls: str,
        metadata: str,
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        batch_size: int = 1,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        random_flip: Optional[float] = None,
        video_reshape_mode: str = "center",
        num_workers: int=4,
        epoch_length: int=10000000,
    ):
        super().__init__()
        self.urls = urls
        self.metadata = metadata
        self.max_num_frames = max_num_frames
        self.id_token = id_token
        self.height_buckets = height_buckets
        self.width_buckets = width_buckets
        self.frame_buckets = frame_buckets
        self.random_flip = random_flip
        self.video_reshape_mode = video_reshape_mode
        self.num_workers = num_workers
        self.epoch_length = epoch_length
        self.batch_size = batch_size


    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = WDSVideoDataset(self.urls,
                                            self.metadata,
                                            self.max_num_frames,
                                            self.id_token,
                                            self.height_buckets,
                                            self.width_buckets,
                                            self.frame_buckets,
                                            self.random_flip,
                                            self.video_reshape_mode).get_dataset()

    def train_dataloader(self):
        loader = wds.WebLoader(self.dataset, 
                               batch_size=None, 
                               shuffle=False, 
                               num_workers=self.num_workers,
                               persistent_workers=False,
                               pin_memory=False)
        # Unbatch, shuffle between workers, then rebatch.
        loader = loader.unbatched().shuffle(10).batched(self.batch_size)
        # Since we are using resampling, the dataset is infinite; set an artificial epoch size.
        loader = loader.with_epoch(self.epoch_length)
        return loader
