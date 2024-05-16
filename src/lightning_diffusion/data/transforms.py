import torch
import torchvision
from torchvision.transforms.functional import crop
from PIL import Image
import random

class RandomCropWithInfo(torch.nn.Module):
    def __init__(self,
                 *args,
                 size: int,
                 **kwargs) -> None:
        super().__init__()
        self.size = size
        self.transform = torchvision.transforms.RandomCrop(
            *args, size, **kwargs)
        
    def forward(self, image: Image):
        size_info = {}
        size_info["before_crop_size"] = [image.height, image.width]

        y1, x1, h, w = self.transform.get_params(image, (self.size, self.size))
        image = crop(image, y1, x1, h, w)
        size_info["crop_top_left"] = [y1, x1]
        size_info["crop_bottom_right"] = [y1 + h, x1 + w]

        return image, size_info

class ComputeTimeIds(torch.nn.Module):
    """Compute time ids as 'time_ids'"""

    def forward(self, image: Image, size_info: dict) -> dict | tuple[list, list] | None:
        assert "original_img_shape" in size_info
        assert "crop_top_left" in size_info

        ori_img_shape = size_info["original_img_shape"]
        crop_top_left = size_info["crop_top_left"]

        target_size = [image.height, image.width]
        time_ids = ori_img_shape + crop_top_left + target_size

        return torch.tensor(time_ids)