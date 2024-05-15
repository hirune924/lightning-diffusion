from lightning.pytorch.cli import LightningCLI
# simple demo classes for your convenience
from lightning_diffusion.data.datasets import HFDataModule
from lightning_diffusion.model.stable_diffusion import StableDiffusionModule
from lightning_diffusion.model.stable_diffusion_xl import StableDiffusionXLModule
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(subclass_mode_data=True, subclass_mode_model=True,
                       parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    cli_main()