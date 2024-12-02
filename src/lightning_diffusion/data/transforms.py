import torch
import torchvision
from torchvision.transforms.functional import crop
from PIL import Image, ImageDraw
import random
from bs4 import BeautifulSoup
import ftfy
import re
import urllib.parse as ul
import html
import numpy as np
import math
import cv2
from torchvision.transforms import v2
Image.MAX_IMAGE_PIXELS = 1000000000

ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],
}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }

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

class MultiAspectRatioResizeCenterCropWithInfo(torch.nn.Module):
    def __init__(self,
                 *args,
                 sizes: list[list[int]] | dict[str, list[int]],
                 **kwargs) -> None:
        super().__init__()
        if isinstance(sizes, dict):
            self.sizes = list(sizes.values())
        else:
            self.sizes = sizes
        self.aspect_ratios = np.array([s[0] / s[1] for s in sizes])
        #self.transform = torchvision.transforms.RandomCrop(
        #    *args, size, **kwargs)
        
    def forward(self, image: Image.Image | list[Image.Image]):
        if isinstance(image, list):
            image_list = image
        else:
            image_list = [image]

        size_info = {}

        for i, img in enumerate(image_list):
            aspect_ratio = img.height / img.width
            bucket_id = np.argmin(np.abs(aspect_ratio - self.aspect_ratios))
            # Resize the image to fit the target size while maintaining aspect ratio
            target_size = self.sizes[bucket_id]
            scale = max(target_size[0] / img.height, target_size[1] / img.width)
            new_size = (int(img.width * scale), int(img.height * scale))
            resized_image = v2.functional.resize(img, new_size, antialias=True)
            # Center crop to the target size
            cropped_image = v2.functional.center_crop(resized_image, target_size)

            image_list[i] = cropped_image

        size_info["before_crop_size"] = [resized_image.height, resized_image.width]
        size_info["crop_top_left"] = [
            (resized_image.height - target_size[0]) // 2,
            (resized_image.width - target_size[1]) // 2
        ]
        size_info["crop_bottom_right"] = [
            size_info["crop_top_left"][0] + target_size[0],
            size_info["crop_top_left"][1] + target_size[1]
        ]
        bucket_id = str(np.argmin(np.abs(aspect_ratio - self.aspect_ratios)))

        if isinstance(image, list):
            image = image_list
        else:
            image = image_list[0]

        return image, bucket_id, size_info

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
    
class T5TextPreprocess(torch.nn.Module):
    """T5 Text Preprocess.'"""
    def __init__(self, clean_caption: bool = True) -> None:
        super().__init__()
        self.clean_caption = clean_caption
        self.bad_punct_regex = re.compile(
            r"["  # noqa
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + "\\"
            + r"\/"
            + r"\*"
            + r"]{1,}",
        )

    def _clean_caption(self, caption: str) -> str:  # noqa
        """Clean caption.

        Copied from
        diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
        """
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)  # noqa
        caption = re.sub(r"[‘’]", "'", caption)  # noqa

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip addresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(self.bad_punct_regex, r" ", caption)
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:  # noqa
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "",
            caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        # j2d1a2a...
        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)  # noqa

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()
    
    def forward(self, text: str) -> str:
        if self.clean_caption:
            text = self._clean_caption(text)
        else:
            text = text.lower().strip()

        return text
    
class GenerateRandomMask(torch.nn.Module):
    """Generate Random Mask.'"""
    def __init__(self, mask_mode: str = "bbox", 
                 mask_config: dict = {}) -> None:
        super().__init__()
        self.mask_mode = mask_mode
        self.mask_config = mask_config
    
    def forward(self, mask_size: tuple[int, int]) -> np.ndarray:
        """Transform function.

        Args:
        ----
            mask_size: size of mask (height, width).

        Returns:
        -------
            dict: A dict containing the processed data and information.
        """
        if self.mask_mode == "bbox":
            mask_bbox = random_bbox(img_shape=mask_size,
                                    **self.mask_config)
            mask = bbox2mask(mask_size, mask_bbox)
        elif self.mask_mode == "irregular":
            mask = get_irregular_mask(img_shape=mask_size,
                                      **self.mask_config)
        elif self.mask_mode == "ff":
            mask = brush_stroke_mask(img_shape=mask_size,
                                     **self.mask_config)
        elif self.mask_mode == "whole":
            mask = np.ones(mask_size[:2], dtype=np.uint8)[:, :, None]
        else:
            msg = f"Mask mode {self.mask_mode} has not been implemented."
            raise NotImplementedError(msg)
        return mask
    
def random_bbox(img_shape: tuple[int, int],
                max_bbox_shape: int | tuple[int, int],
                max_bbox_delta: int | tuple[int, int] = 40,
                min_margin: int | tuple[int, int] = 20,
                ) -> tuple[int, int, int, int]:
    """Generate a random bbox for the mask on a given image.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/main/mmagic/utils/trans_utils.py

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
    ----
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
    -------
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)

    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        msg = (f"mask shape {max_bbox_shape} should be smaller than image"
               f" shape {img_shape}")
        raise ValueError(msg)
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        msg = (f"mask delta {max_bbox_delta} should be smaller thanmask"
               f" shape {max_bbox_shape}")
        raise ValueError(msg)
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        msg = (f"Margin {min_margin} cannot be satisfied for imgshape"
               f" {img_shape} and mask shape {max_bbox_shape}")
        raise ValueError(msg)

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def bbox2mask(img_shape: tuple[int, int],
              bbox: tuple[int, int, int, int],
              dtype: str = "uint8") -> np.ndarray:
    """Generate mask in np.ndarray from bbox.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/main/mmagic/utils/trans_utils.py

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
    ----
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        np.dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'

    Returns:
    -------
        mask (np.ndarray): Mask in the shape of (h, w, 1).
    """
    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask


def random_irregular_mask(img_shape: tuple[int, int],
                          num_vertices: int | tuple[int, int] = (4, 8),
                          max_angle: float = 4,
                          length_range: int | tuple[int, int] = (10, 100),
                          brush_width: int | tuple[int, int] = (10, 40),
                          dtype: str = "uint8") -> np.ndarray:
    """Generate random irregular masks.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/main/mmagic/utils/trans_utils.py

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
    ----
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        np.dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'

    Returns:
    -------
        mask (np.ndarray): Mask in the shape of (h, w, 1).
    """
    h, w = img_shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        msg = (f"The type of length_range should be intor tuple[int], but"
               f" got type: {length_range}")
        raise TypeError(msg)
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        msg = (f"The type of num_vertices should be intor tuple[int], but"
               f" got type: {num_vertices}")
        raise TypeError(msg)

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        msg = (f"The type of brush_width should be intor tuple[int], "
               f"but got type: {brush_width}")
        raise TypeError(msg)

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    return np.expand_dims(mask, axis=2)



def get_irregular_mask(img_shape: tuple[int, int],
                       area_ratio_range: tuple[float, float] = (0.15, 0.5),
                       **kwargs) -> np.ndarray:
    """Get irregular mask with the constraints in mask ratio.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/main/mmagic/utils/trans_utils.py

    Args:
    ----
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
    -------
        mask (np.ndarray): Mask in the shape of (h, w, 1).
    """
    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask


def brush_stroke_mask(img_shape: tuple[int, int],
                      num_vertices: int | tuple[int, int] = (4, 12),
                      mean_angle: float = 2 * math.pi / 5,
                      angle_range: float = 2 * math.pi / 15,
                      brush_width: int | tuple[int, int] = (12, 40),
                      max_loops: int = 4,
                      dtype: str = "uint8") -> np.ndarray:
    """Generate free-form mask.

    Copied from
    https://github.com/open-mmlab/mmagic/blob/main/mmagic/utils/trans_utils.py

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
    ----
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
            Default: 4.
        np.dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
    -------
        mask (np.ndarray): Mask in the shape of (h, w, 1).
    """
    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        msg = (f"The type of num_vertices should be intor tuple[int], but"
               f" got type: {num_vertices}")
        raise TypeError(msg)

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        msg = (f"The type of brush_width should be intor tuple[int], but"
               f" got type: {brush_width}")
        raise TypeError(msg)

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new("L", (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    return mask[:, :, None]

def XDoG_filter(image,
                kernel_size=0,
                sigma=1.4,
                k_sigma=1.6,
                epsilon=0,
                phi=10,
                gamma=0.98):
    """XDoG(Extended Difference of Gaussians)を処理した画像を返す

    Args:
        image: OpenCV Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        eps: threshold value between dark and bright
        phi: soft threshold
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the XDoG.
    """
    epsilon /= 255
    dog = DoG_filter(image, kernel_size, sigma, k_sigma, gamma)
    dog /= dog.max()
    e = 1 + np.tanh(phi * (dog - epsilon))
    e[e >= 1] = 1
    return e.astype('uint8') * 255


def DoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, gamma=1):
    """DoG(Difference of Gaussians)を処理した画像を返す

    Args:
        image: OpenCV Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the DoG.
    """
    g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
    return g1 - gamma * g2


def draw_colored_point(line_img, color_img, x, y, radius, alpha=0.9, color_mode="median"):
    color_img = color_img.convert("RGB").resize(line_img.size)

    # Convert PIL Images to numpy arrays for processing
    line_img_cv = np.array(line_img)
    color_img_cv = np.array(color_img)

    # Get the average color from the color image at circle location
    circle_mask = np.zeros(line_img_cv.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), radius, 255, -1)
    circle_region = color_img_cv[circle_mask == 255]
    avg_color = tuple(map(int, np.median(circle_region, axis=0)))

    # Draw the circle with the average color from color image
    # Create a transparent circle mask
    overlay_mask = np.zeros_like(line_img_cv, dtype=np.uint8)
    cv2.circle(overlay_mask, (x, y), radius, (255,255,255), -1)
    
    # Create circle overlay with the color
    circle_overlay = np.zeros_like(line_img_cv, dtype=np.uint8)
    cv2.circle(circle_overlay, (x, y), radius, avg_color, -1)
    
    # Blend only within the circle area
    mask_bool = (overlay_mask > 0)
    if color_mode == "median":
        line_img_cv[mask_bool] = cv2.addWeighted(line_img_cv, 1-alpha, circle_overlay, alpha, 0)[mask_bool]
    elif color_mode == "swap":
        line_img_cv[mask_bool] = color_img_cv[mask_bool]

    # Convert back to PIL Image and return
    return Image.fromarray(line_img_cv)

def draw_random_colored_points(line_img, color_img, num_range=(1,8), radius_range=(16,40), color_mode="median"):
    # Generate random number of circles based on input range
    num_circles = np.random.randint(num_range[0], num_range[1] + 1)
    
    # Make a copy of line_img to avoid modifying original
    result_img = line_img.copy()
    
    for _ in range(num_circles):
        # Random coordinates within image bounds
        x = np.random.randint(0, line_img.size[0])
        y = np.random.randint(0, line_img.size[1])
        
        # Random radius between given range
        radius = np.random.randint(radius_range[0], radius_range[1] + 1)
        
        # Draw the colored point
        result_img = draw_colored_point(result_img, color_img, x, y, radius, color_mode=color_mode)
    
    return result_img