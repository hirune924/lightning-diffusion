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
        
    def forward(self, image: Image.Image | list[Image.Image]):
        if isinstance(image, list):
            image_list = image
        else:
            image_list = [image]

        size_info = {}
        size_info["before_crop_size"] = [image_list[0].height, image_list[0].width]

        y1, x1, h, w = self.transform.get_params(image_list[0], (self.size, self.size))
        image_list = [crop(im, y1, x1, h, w) for im in image_list]
        size_info["crop_top_left"] = [y1, x1]
        size_info["crop_bottom_right"] = [y1 + h, x1 + w]

        if isinstance(image, list):
            image = image_list
        else:
            image = image_list[0]

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