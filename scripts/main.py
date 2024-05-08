from lightning.pytorch.cli import LightningCLI
# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from lightning_diffusion.data.datasets import HFDataModule

def cli_main():
    cli = LightningCLI(DemoModel, HFDataModule)

if __name__ == "__main__":
    cli_main()