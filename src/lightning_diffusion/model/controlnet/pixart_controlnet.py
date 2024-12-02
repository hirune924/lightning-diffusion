from typing import Any
import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, PixArtTransformer2DModel, PixArtAlphaPipeline, PixArtSigmaPipeline, StableDiffusionPipeline
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig
from lightning_diffusion.model.text_to_image.utils.pixart import discretized_gaussian_log_likelihood
from lightning_diffusion.model.controlnet.pipelines.pipeline_pixart_alpha_controlnet import PixArtAlphaControlNetPipeline
from lightning_diffusion.model.controlnet.pipelines.pipeline_pixart_sigma_controlnet import PixArtSigmaControlNetPipeline
from lightning_diffusion.model.controlnet.archs.controlnet_pixart import PixArtTransformer2DControlNetModel, PixArtTransformer2DControlNetNoVAEModel
from lightning_diffusion.model.controlnet.archs.pixart_transformer_2d_controlnet import PixArtTransformer2DControlNet
from PIL import Image
from diffusers.utils import load_image
from lightning_diffusion.utils.utils import load_pytorch_model

class PixArtControlnetModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 base_transformer: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 controlnet_weight: str = None,
                 gradient_checkpointing: bool = True,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 tokenizer_max_length: int = 120,
                 noise_offset: float = 0.0,
                 use_resolution: bool = False,
                 enable_vb_loss: bool = True,
                 no_vae: bool = False):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.enable_vb_loss = enable_vb_loss
        self.tokenizer_max_length = tokenizer_max_length
        self.use_resolution = use_resolution
        self.base_model = base_model
        self.no_vae = no_vae
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model, subfolder="vae")
        self.transformer = PixArtTransformer2DControlNet.from_pretrained(pretrained_model_name_or_path=base_transformer,
                                                              use_additional_conditions=use_resolution,
                                                              subfolder="transformer")
        if self.no_vae:
            self.controlnet = PixArtTransformer2DControlNetNoVAEModel.from_transformer(self.transformer, num_layers=13)
        else:
            self.controlnet = PixArtTransformer2DControlNetModel.from_transformer(self.transformer, num_layers=13)
        if controlnet_weight is not None:
            self.controlnet = load_pytorch_model(ckpt_name=controlnet_weight, model=self.controlnet, ignore_suffix="controlnet")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(True)
        self.controlnet.adaln_single.requires_grad_(False)
        self.controlnet.caption_projection.requires_grad_(False)
        self.controlnet.pos_embed.requires_grad_(False)

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()

        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=1)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              condition_image: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int | None = 512,
              width: int | None = 512,
              num_inference_steps: int = 20,
              ) -> list[np.ndarray]:
        
        if self.tokenizer_max_length == 120:
            pipe_cls = PixArtAlphaControlNetPipeline
        else:
            pipe_cls = PixArtSigmaControlNetPipeline
        pipeline = pipe_cls(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_name_or_path=self.base_model, subfolder="scheduler"),
            controlnet=self.controlnet
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for i, (p, img) in enumerate(zip(prompt, condition_image, strict=True)):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            generator = torch.Generator(device=self.device).manual_seed(i)
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
                control_image=pil_img,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type="pil",
                generator=generator
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
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        batch["text"] = text_inputs.input_ids.to(self.device)
        batch["attention_mask"] = text_inputs.attention_mask.to(self.device)
        latents = self.vae.encode(batch["image"]).latent_dist.sample() * self.vae.config.scaling_factor
        if not self.no_vae:
            control_img = self.vae.encode(batch["condition_img"]).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            control_img = batch["condition_img"]

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
        # if use 1024 alpha model, use additional conditions.
        #if self.transformer.config.sample_size == 128:  # noqa
        if self.use_resolution:
            added_cond_kwargs = {"resolution": batch["resolution"],
                                 "aspect_ratio": batch["aspect_ratio"]}
        else:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        controlnet_block_samples = self.controlnet(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch["attention_mask"],
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
            controlnet_cond=control_img,
            return_dict=False,
        )[0]

        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch["attention_mask"],
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
            controlnet_block_samples=controlnet_block_samples).sample
        
        loss_dict = self.loss(model_pred, noise, latents, timesteps, noisy_latents)
        total_loss = torch.sum(torch.stack(list(loss_dict.values())))
        self.log_dict(loss_dict)
        self.log("train_loss", total_loss)
        return total_loss
    
    def loss(self,
             model_pred: torch.Tensor,
             noise: torch.Tensor,
             latents: torch.Tensor,
             timesteps: torch.Tensor,
             noisy_model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        loss_dict = {}

        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_pred, model_var_values = model_pred.chunk(2, dim=1)

        if self.enable_vb_loss:
            alphas_cumprod = self.scheduler.alphas_cumprod
            alphas = self.scheduler.alphas.to(self.device)
            betas = self.scheduler.betas.to(self.device)
            alphas_cumprod_prev = torch.cat([torch.tensor([self.scheduler.one], device=self.device),
                                        self.scheduler.alphas_cumprod[:-1]])
            alpha_prod_t = alphas_cumprod[
                timesteps][...,None,None,None]
            alpha_prod_t_prev = alphas_cumprod_prev[
                timesteps][...,None,None,None]
            current_beta_t = betas[timesteps][...,None,None,None]
            current_alpha_t = alphas[timesteps][...,None,None,None]
            variance = (1 - alpha_prod_t_prev
                        ) / (1 - alpha_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)

            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (model_var_values.float() + 1) / 2
            model_log_variance = frac * max_log  + (1 - frac) * min_log
            true_log_variance = torch.log(variance)

            pred_x0 = torch.sqrt(
                1.0 / alpha_prod_t,
            ) * noisy_model_input.float() - torch.sqrt(
                1.0 / alpha_prod_t - 1) * model_pred.float()

            posterior_mean_coef1 = (
                current_beta_t * torch.sqrt(
                    alpha_prod_t_prev) / (1.0 - alpha_prod_t)
            )
            posterior_mean_coef2 = (
                (1.0 - alpha_prod_t_prev) * torch.sqrt(
                    current_alpha_t) / (1.0 - alpha_prod_t)
            )
            model_mean = (
                posterior_mean_coef1 * pred_x0.float(
                    ) + posterior_mean_coef2 * noisy_model_input.float()
            )
            true_mean = (
                posterior_mean_coef1 * latents.float(
                    ) + posterior_mean_coef2 * noisy_model_input.float()
            )

            kl = 0.5 * (
                -1.0
                + model_log_variance
                - true_log_variance
                + torch.exp(true_log_variance - model_log_variance)
                + ((true_mean - model_mean) ** 2) * torch.exp(-model_log_variance)
            )
            kl = kl.mean(dim=[1, 2, 3]) / np.log(2.0)

            decoder_nll = - discretized_gaussian_log_likelihood(
                latents.float(), means=model_mean, log_scales=0.5 * model_log_variance,
            )
            decoder_nll = decoder_nll.mean(dim=[1, 2, 3]) / np.log(2.0)

            # At the first timestep return the decoder NLL,
            # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            vb_loss = torch.where(timesteps == 0, decoder_nll, kl)
            loss_dict["vb_loss"] = vb_loss.mean()

        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)
        
        loss_dict["l2_loss"] = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")

        return loss_dict
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.controlnet.save_pretrained(f'{self.trainer.default_root_dir}/controlnet_weight/step_{self.global_step}')
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("controlnet."):
                del checkpoint['state_dict'][k]
        

class PixArtControlnetLoRAModule(PixArtControlnetModule):
    def __init__(self, 
                 base_model: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 base_transformer: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 controlnet_weight: str = None,
                 gradient_checkpointing: bool = True,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 tokenizer_max_length: int = 120,
                 noise_offset: float = 0.0,
                 use_resolution: bool = False,
                 enable_vb_loss: bool = True,
                 no_vae: bool = False,
                 lora_rank: int = 8):
        super(PixArtControlnetModule, self).__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.enable_vb_loss = enable_vb_loss
        self.tokenizer_max_length = tokenizer_max_length
        self.use_resolution = use_resolution
        self.base_model = base_model
        self.no_vae = no_vae
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model, subfolder="vae")
        self.transformer = PixArtTransformer2DControlNet.from_pretrained(pretrained_model_name_or_path=base_transformer,
                                                              use_additional_conditions=use_resolution,
                                                              subfolder="transformer")
        if self.no_vae:
            self.controlnet = PixArtTransformer2DControlNetNoVAEModel.from_transformer(self.transformer, num_layers=13)
        else:
            self.controlnet = PixArtTransformer2DControlNetModel.from_transformer(self.transformer, num_layers=13)
        if controlnet_weight is not None:
            self.controlnet = load_pytorch_model(ckpt_name=controlnet_weight, model=self.controlnet, ignore_suffix="controlnet")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.controlnet.adaln_single.requires_grad_(False)
        self.controlnet.caption_projection.requires_grad_(False)
        self.controlnet.pos_embed.requires_grad_(False)

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()

        lora_config = LoraConfig(r=lora_rank,
                                    lora_alpha=lora_rank,
                                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()
        self.train()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict
        lora_state_dict = get_peft_model_state_dict(self.transformer, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(
                f'{self.trainer.default_root_dir}/lora_weight/step_{self.global_step}',
                unet_lora_layers=lora_state_dict
        )
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("transformer."):
                del checkpoint['state_dict'][k]

class PixArtControlnetFTModule(PixArtControlnetModule):
    def __init__(self, 
                 base_model: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 base_transformer: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 controlnet_weight: str = None,
                 gradient_checkpointing: bool = True,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 tokenizer_max_length: int = 120,
                 noise_offset: float = 0.0,
                 use_resolution: bool = False,
                 enable_vb_loss: bool = True,
                 no_vae: bool = False):
        super(PixArtControlnetModule, self).__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.enable_vb_loss = enable_vb_loss
        self.tokenizer_max_length = tokenizer_max_length
        self.use_resolution = use_resolution
        self.base_model = base_model
        self.no_vae = no_vae
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model, subfolder="vae")
        self.transformer = PixArtTransformer2DControlNet.from_pretrained(pretrained_model_name_or_path=base_transformer,
                                                              use_additional_conditions=use_resolution,
                                                              subfolder="transformer")
        if self.no_vae:
            self.controlnet = PixArtTransformer2DControlNetNoVAEModel.from_transformer(self.transformer, num_layers=13)
        else:
            self.controlnet = PixArtTransformer2DControlNetModel.from_transformer(self.transformer, num_layers=13)
        if controlnet_weight is not None:
            self.controlnet = load_pytorch_model(ckpt_name=controlnet_weight, model=self.controlnet, ignore_suffix="controlnet")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        self.controlnet.requires_grad_(False)
        self.controlnet.adaln_single.requires_grad_(False)
        self.controlnet.caption_projection.requires_grad_(False)
        self.controlnet.pos_embed.requires_grad_(False)

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.controlnet.enable_gradient_checkpointing()

        self.train()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=1)

        return [optimizer]#, [{'scheduler': scheduler, 'interval': 'step'}]
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.transformer.save_pretrained(f'{self.trainer.default_root_dir}/transformer_weight/step_{self.global_step}')
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("transformer."):
                del checkpoint['state_dict'][k]
