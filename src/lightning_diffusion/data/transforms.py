import torch
import torchvision
from torchvision.transforms.functional import crop
from PIL import Image
import random
from bs4 import BeautifulSoup
import ftfy
import re
import urllib.parse as ul
import html

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
    