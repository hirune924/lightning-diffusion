from lightning.pytorch.cli import LightningCLI
# simple demo classes for your convenience
from lightning_diffusion.data.datasets import HFDataModule
from lightning_diffusion.model.stable_diffusion import StableDiffusionModel
import torch
torch.set_float32_matmul_precision('medium')
def cli_main():
    cli = LightningCLI(StableDiffusionModel, HFDataModule, subclass_mode_data=False, subclass_mode_model=False,
                       parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    cli_main()