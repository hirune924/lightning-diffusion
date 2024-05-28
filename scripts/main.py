from lightning.pytorch.cli import LightningCLI
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(subclass_mode_data=True, subclass_mode_model=True,
                       parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    cli_main()