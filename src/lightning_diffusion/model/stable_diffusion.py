import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig

class StableDiffusionModel(L.LightningModule):
    def __init__(self, 
                 base_model: str = "runwayml/stable-diffusion-v1-5", 
                 train_mode: str = "unet_attn",
                 gradient_checkpointing: bool = False,
                 cfg_prob: float = 0.1,
                 input_perturbation_gamma: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.cfg_prob = cfg_prob
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

        if train_mode == "unet_all":
            self.unet.requires_grad_(True)

        elif train_mode == "unet_attn":
            for name, param in self.unet.named_parameters():
                if "to_k" in name or "to_v" in name:
                    param.requires_grad = True

        elif train_mode == "unet_lora":
            lora_config = LoraConfig(r=8,
                                     lora_alpha=8,
                                     target_modules=["to_q", "to_v", "to_k", "to_out.0"])
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()

        elif train_mode == "text_encoder_lora":
            lora_config = LoraConfig(r=8,
                                     lora_alpha=8,
                                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            self.text_encoder.print_trainable_parameters()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()
        self.train()
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
        batch["text"] = ["" if np.random.rand() < self.cfg_prob else t for t in batch["text"]]
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

        if self.input_perturbation_gamma > 0:
            noise = noise + self.input_perturbation_gamma * torch.randn_like(noise)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        else:
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["text"], return_dict=True).last_hidden_state

        noise_pred = self.unet(
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
        
        loss = F.mse_loss(noise_pred.float(), gt.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss
        