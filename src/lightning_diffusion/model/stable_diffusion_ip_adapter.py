import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.embeddings import ImageProjection, IPAdapterPlusImageProjection, MultiIPAdapterImageProjection
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import numpy as np
from peft import get_peft_model, LoraConfig
from lightning_diffusion.model.utils.ip_adapter import set_unet_ip_adapter, process_ip_adapter_state_dict
from PIL import Image
from diffusers.utils import load_image
from transformers import AutoProcessor, CLIPImageProcessor

class StableDiffusionIPAdapterModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "runwayml/stable-diffusion-v1-5", 
                 image_encoder: str = "openai/clip-vit-large-patch14",
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.image_encoder_name = image_encoder
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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder)
        self.image_projection = ImageProjection(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            image_embed_dim=self.image_encoder.config.projection_dim,
            num_image_text_embeds=4,
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.image_projection.requires_grad_(True)

        self.unet.requires_grad_(requires_grad=False)
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
              height: int | None = 512,
              width: int | None = 512,
              num_inference_steps: int = 50,
              ) -> list[np.ndarray]:
        
        orig_encoder_hid_proj = self.unet.encoder_hid_proj
        orig_encoder_hid_dim_type = self.unet.config.encoder_hid_dim_type

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            image_encoder=self.image_encoder,
            feature_extractor=CLIPImageProcessor.from_pretrained(self.image_encoder_name),
            safety_checker=None,
            requires_safety_checker=False
        )

        adapter_state_dict = process_ip_adapter_state_dict(
            self.unet, self.image_projection)
        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layer = (
            pipeline.unet._convert_ip_adapter_image_proj_to_diffusers(  # noqa
                adapter_state_dict["image_proj"]))
        image_projection_layer.to(
            device=pipeline.unet.device, dtype=pipeline.unet.dtype)

        pipeline.unet.encoder_hid_proj = MultiIPAdapterImageProjection(
            [image_projection_layer])
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

        del pipeline, adapter_state_dict
        torch.cuda.empty_cache()

        self.unet.encoder_hid_proj = orig_encoder_hid_proj
        self.unet.config.encoder_hid_dim_type = orig_encoder_hid_dim_type

        return images

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["image"])
        if self.ucg_rate > 0:
            batch["text"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["text"]]
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
        image_embeds = self.image_encoder(batch["clip_img"]).image_embeds
        # random zeros image embeddings
        mask = torch.multinomial(
            torch.Tensor([
                self.ucg_rate,
                1 - self.ucg_rate,
            ]),
            len(image_embeds),
            replacement=True).to(image_embeds)
        image_embeds = (image_embeds * mask.view(-1, 1)).view(num_batches, 1, 1, -1)

        ip_tokens = self.image_projection(image_embeds)
        encoder_hidden_states = (encoder_hidden_states, [ip_tokens])

        model_pred = self.unet(
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
        

class StableDiffusionIPAdapterPlusModule(StableDiffusionIPAdapterModule):
    def __init__(self, 
                 base_model: str = "runwayml/stable-diffusion-v1-5", 
                 image_encoder: str = "openai/clip-vit-large-patch14",
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 noise_offset: float = 0.0):
        super(StableDiffusionIPAdapterModule, self).__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.image_encoder_name = image_encoder
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
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.image_projection.requires_grad_(True)

        self.unet.requires_grad_(requires_grad=False)
        set_unet_ip_adapter(self.unet, num_tokens=16)

        self.train()

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["image"])
        if self.ucg_rate > 0:
            batch["text"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["text"]]
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
        # random zeros image
        clip_img = batch["clip_img"]
        mask = torch.multinomial(
            torch.Tensor([
                self.ucg_rate,
                1 - self.ucg_rate,
            ]),
            len(clip_img),
            replacement=True).to(clip_img)
        clip_img = clip_img * mask.view(-1, 1, 1, 1)

        image_embeds = self.image_encoder(batch["clip_img"], output_hidden_states=True).hidden_states[-2]

        ip_tokens = self.image_projection(image_embeds)

        encoder_hidden_states = (encoder_hidden_states, [ip_tokens])

        model_pred = self.unet(
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
        

