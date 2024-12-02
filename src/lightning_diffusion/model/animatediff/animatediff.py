from typing import Any
import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, AnimateDiffPipeline, UNetMotionModel
from diffusers.models.unets.unet_motion_model import MotionAdapter
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig
from einops import rearrange
import bitsandbytes as bnb

class AnimateDiffModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "runwayml/stable-diffusion-v1-5", 
                 motion_module_model: str = "guoyww/animatediff-motion-adapter-v1-5-2",
                 gradient_checkpointing: bool = False,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.base_model = base_model
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="unet")
        motion_module = MotionAdapter.from_pretrained(motion_module_model)
        self.motion_unet = UNetMotionModel.from_unet2d(unet, motion_module)
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.motion_unet.requires_grad_(False)
        for name, param in self.motion_unet.named_parameters():
            if "motion_modules" in name:
                param.requires_grad = True

        if gradient_checkpointing:
            self.motion_unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()
        self.train()

    def configure_optimizers(self):
        optimizer = bnb.optim.AdamW8bit(list(filter(lambda p: p.requires_grad, self.parameters())), lr=1.0e-5, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int | None = 512,
              width: int | None = 512,
              num_frames: int = 24,
              num_inference_steps: int = 50,
              ) -> list[np.ndarray]:
        pipeline = AnimateDiffPipeline.from_pretrained(
            self.base_model,
            unet=self.motion_unet
        ).to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        results = []
        for p in prompt:
            vid = pipeline(
                p,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                num_frames=num_frames,
                output_type="pil"
                ).frames[0]

            #vid = vid[0].permute(1, 2, 3, 0).cpu().numpy()  # (f, h, w, c)
            #vid = (vid * 255).astype(np.uint8)
            results.append(vid)

        del pipeline
        torch.cuda.empty_cache()

        return results

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["pixel_values"])
        if self.ucg_rate > 0:
            batch["text"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["text"]]
        batch["text"] = self.tokenizer(
            batch["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        pixel_values = batch["pixel_values"]
        video_length = pixel_values.shape[1]
        with torch.no_grad():
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0,self.scheduler.config.num_train_timesteps, (num_batches, ),
            dtype=torch.int64, device=self.device)

        if self.noise_offset > 0:
            noise += self.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
        if self.input_perturbation_gamma > 0:
            new_noise = noise + self.input_perturbation_gamma * torch.randn_like(noise)
            noisy_latents = self.scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["text"], return_dict=False)[0]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=video_length, dim=0)

        model_pred = self.motion_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample
        
        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)
        
        loss = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss
    
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.motion_unet.save_motion_modules(f'{self.trainer.default_root_dir}/motion_module_weight/step_{self.global_step}')
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if "motion_modules" not in k:
                del checkpoint['state_dict'][k]