import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel, PixArtAlphaPipeline
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig

class PixArtAlphaModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "PixArt-alpha/PixArt-XL-2-512x512", 
                 train_mode: str = "transformer_lora",
                 gradient_checkpointing: bool = True,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path="stabilityai/sd-vae-ft-ema")
        self.transformer = Transformer2DModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="transformer")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if train_mode == "transformer":
            self.transformer.requires_grad_(True)

        elif train_mode == "transformer_lora":
            lora_config = LoraConfig(r=8,
                                     lora_alpha=8,
                                     target_modules=["to_q", "to_v", "to_k", "to_out.0"])
            self.transformer = get_peft_model(self.transformer, lora_config)
            self.transformer.print_trainable_parameters()

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

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
        
        pipeline = PixArtAlphaPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.scheduler
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
        if self.ucg_rate > 0:
            batch["text"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["text"]]
        text_inputs = self.tokenizer(
            batch["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        batch["text"] = text_inputs.input_ids.to(self.device)
        batch["attention_mask"] = text_inputs.attention_mask.to(self.device)
        
        latents = self.vae.encode(batch["image"]).latent_dist.sample() * self.vae.config.scaling_factor
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

        encoder_hidden_states = self.text_encoder(batch["text"], 
                                                  attention_mask=batch["attention_mask"], 
                                                  return_dict=False)[0]
        # input size is 512 or 1024. if use 1024 model, use additional conditions.
        if self.transformer.config.sample_size == 128:  # noqa
            added_cond_kwargs = {"resolution": batch["resolution"],
                                 "aspect_ratio": batch["aspect_ratio"]}
        else:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch["attention_mask"],
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs).sample
        
        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_pred = model_pred.chunk(2, dim=1)[0]

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
        