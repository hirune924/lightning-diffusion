import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

class StableDiffusionModel(L.LightningModule):
    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="unet")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        for name, param in self.unet.named_parameters():
            if 'to_k' in name or 'to_v' in name:
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int | None = 512,
              width: int | None = 512,
              num_inference_steps: int = 50,
              ) -> list[np.ndarray]:
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            feature_extractor=None,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type="pil"
                ).images[0]

            images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["image"])
        batch["text"] = self.tokenizer(
            batch["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        
        latents = self.vae.encode(batch["image"]).latent_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0,self.scheduler.config.num_train_timesteps, (num_batches, ),
            dtype=torch.int64, device=self.device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["text"], return_dict=True).last_hidden_state

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample
        
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss
        