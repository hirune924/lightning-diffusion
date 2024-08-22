import torch
from loguru import logger

####################
# Utils
####################
def load_pytorch_model(ckpt_name: str, model: torch.nn.Module, ignore_suffix: str = "model", only_target: bool = True) -> torch.nn.Module:
    ckpt = torch.load(ckpt_name, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix) + "."):
            name = name.replace(str(ignore_suffix) + ".", "", 1)  # remove `model.`
            if only_target:
                new_state_dict[name] = v
        if not only_target:
            new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    logger.info(res)
    return model
