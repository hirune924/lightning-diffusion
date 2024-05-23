import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.models.embeddings import ImageProjection, IPAdapterPlusImageProjection, MultiIPAdapterImageProjection
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer, CLIPVisionModelWithProjection
import numpy as np
from peft import get_peft_model, LoraConfig
from lightning_diffusion.model.utils.ip_adapter import set_unet_ip_adapter
from transformers import AutoProcessor, CLIPImageProcessor
from diffusers.utils import load_image
from PIL import Image

class StableDiffusionXLIPAdapterModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0", 
                 image_encoder: str = "openai/clip-vit-large-patch14",
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.image_encoder_name = image_encoder
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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder)
        self.image_projection = ImageProjection(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            image_embed_dim=self.image_encoder.config.projection_dim,
            num_image_text_embeds=4,
        )
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.image_projection.requires_grad_(True)

        set_unet_ip_adapter(self.unet, num_tokens=4)

        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              example_image: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int | None = 1024,
              width: int | None = 1024,
              num_inference_steps: int = 20,
              ) -> list[np.ndarray]:
        orig_encoder_hid_proj = self.unet.encoder_hid_proj
        orig_encoder_hid_dim_type = self.unet.config.encoder_hid_dim_type
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            image_encoder=self.image_encoder,
            feature_extractor=CLIPImageProcessor.from_pretrained(self.image_encoder_name),
            scheduler=self.scheduler,
            force_zeros_for_empty_prompt=True,
            add_watermarker=None
        )
        pipeline.unet.encoder_hid_proj = MultiIPAdapterImageProjection([self.image_projection])
        pipeline.unet.config.encoder_hid_dim_type = "ip_image_proj"

        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, example_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            image = pipeline(
                p,
                ip_adapter_image=pil_img,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type="pil"
                ).images[0]

            images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()
        self.unet.encoder_hid_proj = orig_encoder_hid_proj
        self.unet.config.encoder_hid_dim_type = orig_encoder_hid_dim_type
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
        image_embeds = self.image_encoder(batch["clip_img"]).image_embeds

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

            mask = torch.multinomial(
                torch.Tensor([
                    self.ucg_rate,
                    1 - self.ucg_rate,
                ]),
                num_batches,
                replacement=True).to(self.device)
            image_embeds = (image_embeds * mask.view(-1, 1)).view(num_batches, 1, 1, -1)
        ip_tokens = self.image_projection(image_embeds)

        unet_added_conditions = {
            "time_ids": batch["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            (prompt_embeds, [ip_tokens]),
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


class StableDiffusionXLIPAdapterPlusModule(StableDiffusionXLIPAdapterModule):
    def __init__(self, 
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0", 
                 image_encoder: str = "openai/clip-vit-large-patch14",
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super(StableDiffusionXLIPAdapterModule, self).__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.image_encoder_name = image_encoder
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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder)
        self.image_projection = IPAdapterPlusImageProjection(
            output_dims=self.unet.config.cross_attention_dim,
            embed_dims=self.image_encoder.config.hidden_size,
            hidden_dims=self.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            ffn_ratio=4
        )
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.image_projection.requires_grad_(True)

        set_unet_ip_adapter(self.unet, num_tokens=4)

        self.train()


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
        clip_img = batch["clip_img"]
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

            mask = torch.multinomial(
                torch.Tensor([
                    self.ucg_rate,
                    1 - self.ucg_rate,
                ]),
                num_batches,
                replacement=True).to(self.device)
            clip_img = clip_img * mask.view(-1, 1, 1, 1)
        image_embeds = self.image_encoder(clip_img, output_hidden_states=True).hidden_states[-2]
        ip_tokens = self.image_projection(image_embeds)

        unet_added_conditions = {
            "time_ids": batch["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            (prompt_embeds, [ip_tokens]),
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