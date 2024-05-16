import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig

class StableDiffusionXLModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0", 
                 train_mode: str = "unet_attn",
                 gradient_checkpointing: bool = False,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.tokenizer_one = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                           subfolder="tokenizer", use_fast=False)
        self.tokenizer_two = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                           subfolder="tokenizer_2", use_fast=False)       
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                              subfolder="text_encoder")
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=base_model,
                                                                            subfolder="text_encoder_2")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
                                                 subfolder=None)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="unet")
        
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
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
            self.text_encoder_one = get_peft_model(self.text_encoder_one, lora_config)
            self.text_encoder_two = get_peft_model(self.text_encoder_two, lora_config)
            self.text_encoder_one.print_trainable_parameters()
            self.text_encoder_two.print_trainable_parameters()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder_one.gradient_checkpointing_enable()
            self.text_encoder_two.gradient_checkpointing_enable()
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int | None = 1024,
              width: int | None = 1024,
              num_inference_steps: int = 20,
              ) -> list[np.ndarray]:
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            scheduler=self.scheduler,
            feature_extractor=None,
            force_zeros_for_empty_prompt=True,
            add_watermarker=None
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
        batch["text_one"] = self.tokenizer_one(
            batch["text"],
            max_length=self.tokenizer_one.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        batch["text_two"] = self.tokenizer_two(
            batch["text"],
            max_length=self.tokenizer_two.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)

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

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                batch["text_one"], batch["text_two"])
        
        if self.ucg_rate > 0:
            mask = torch.multinomial(
                torch.Tensor([
                    self.ucg_rate,
                    1 - self.ucg_rate,
                ]),
                num_batches,
                replacement=True).to(self.device)
            prompt_embeds = prompt_embeds * mask.view(-1, 1, 1)
            pooled_prompt_embeds = (pooled_prompt_embeds * mask.view(-1, 1)).view(num_batches, -1)

        unet_added_conditions = {
            "time_ids": batch["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample
        
        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        elif self.scheduler.config.prediction_type == "sample":
            gt = latents
            model_pred = model_pred - noise
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)
        
        loss = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss
        
    def encode_prompt(
        self,
        text_one: torch.Tensor,
        text_two: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt.

        Args:
        ----
            text_one (torch.Tensor): Token ids from tokenizer one.
            text_two (torch.Tensor): Token ids from tokenizer two.

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: Prompt embeddings
        """
        prompt_embeds_list = []

        text_encoders = [self.text_encoder_one, self.text_encoder_two]
        texts = [text_one, text_two]
        for text_encoder, text in zip(text_encoders, texts, strict=True):

            prompt_embeds = text_encoder(
                text,
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the
            # final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
