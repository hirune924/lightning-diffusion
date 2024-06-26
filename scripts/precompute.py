from lightning.pytorch.cli import LightningCLI
import torch
from lightning_diffusion.model import *
from lightning_diffusion.data import *
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
from loguru import logger
from uuid import uuid4
from lightning.fabric import Fabric
from types import MethodType

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wds_save_dir", default=None, type=str, help="")
        parser.add_argument("--repeat", default=1, type=int, help="")
        parser.add_argument("--ckpt_path", default=None, type=str, help="")

@torch.inference_mode(mode=False)
def compute_step(self, batch):
    if self.ucg_rate > 0:
        batch["text"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["text"]]
    batch["text"] = self.tokenizer(
        batch["text"],
        max_length=self.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt").input_ids.to(self.device)
    
    encoder_hidden_states = self.text_encoder(batch["text"], return_dict=False)[0]
    
    latents = self.vae.encode(batch["image"]).latent_dist.sample() * self.vae.config.scaling_factor

    return {'latents': latents.cpu().numpy(), 'encoder_hidden_states': encoder_hidden_states.cpu().numpy()}

def cli_main():
    cli = CustomLightningCLI(subclass_mode_data=True, subclass_mode_model=True,
                       parser_kwargs={"parser_mode": "omegaconf"}, run=False)
    
    logger.info(f"Arguments")
    logger.info(f"wds_save_dir: {cli.config.wds_save_dir}")
    logger.info(f"repeat: {cli.config.repeat}")
    logger.info(f"ckpt_path: {cli.config.ckpt_path}")

    save_dir = Path(cli.config.wds_save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path_pattern = save_dir / "shards-%06d.tar"

    cli.datamodule.setup(stage="fit")
    dataloader = cli.datamodule.train_dataloader()
    model = cli.model
    model.compute_step = MethodType(compute_step, model)

    fabric = Fabric(precision=cli.config.trainer.precision)
    fabric.launch()
    model = fabric.setup(model)
    dataloader = fabric.setup_dataloaders(dataloader)

    # 1e9=1GB
    with wds.ShardWriter(str(save_path_pattern), maxsize=1e9, maxcount=100000) as sink:
        for _ in range(cli.config.repeat):
            for batch in tqdm(dataloader):
                num_batches = len(batch["image"])
                result = model.compute_step(batch)
                for idx in range(num_batches):
                    key = str(uuid4())[:10]
                    sink.write({
                        "__key__": key,
                        "latents.npy": result["latents"][idx],
                        "encoder_hidden_states.npy": result["encoder_hidden_states"][idx],
                    })

if __name__ == "__main__":
    cli_main()