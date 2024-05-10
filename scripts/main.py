from lightning.pytorch.cli import LightningCLI
# simple demo classes for your convenience
from lightning_diffusion.data.datasets import HFDataModule
from lightning_diffusion.model.stable_diffusion import StableDiffusionModel

def cli_main():
    cli = LightningCLI(StableDiffusionModel, HFDataModule)

if __name__ == "__main__":
    cli_main()