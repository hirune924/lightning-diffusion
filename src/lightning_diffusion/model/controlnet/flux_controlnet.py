from typing import Any
import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel, FluxControlNetPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
import numpy as np
from peft import get_peft_model, LoraConfig
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models import FluxControlNetModel
import copy
from PIL import Image
from diffusers.utils import load_image
class FluxControlnetModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "black-forest-labs/FLUX.1-dev", 
                 controlnet_model: str | None = None,
                 gradient_checkpointing: bool = False,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.tokenizer_one = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=None)
        self.tokenizer_two = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2", revision=None)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                              subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.text_encoder_two = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                                            subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae", torch_dtype=torch.bfloat16)
        self.transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="transformer", torch_dtype=torch.bfloat16)
        
        if controlnet_model is not None:
            self.controlnet = FluxControlNetModel.from_pretrained(controlnet_model)
        else:
            self.controlnet = FluxControlNetModel.from_transformer(self.transformer,
                                                                   num_layers=5,
                                                                   num_single_layers=0,
                                                                   num_attention_heads=24,
                                                                   attention_head_dim=128)
        self.controlnet.to(dtype=torch.bfloat16)

        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.transformer.enable_gradient_checkpointing()
            self.text_encoder_one.gradient_checkpointing_enable()
            self.text_encoder_two.gradient_checkpointing_enable()
        self.train()

    def configure_optimizers(self):
        opt_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        optimizer = torch.optim.AdamW(opt_params, lr=self.learning_rate, weight_decay=1e-2)
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
        pipeline = FluxControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
            controlnet=self.controlnet,
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img

            pil_img = pil_img.convert("RGB")
            image = pipeline(
                p,
                control_image=pil_img,
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

        latents = self.vae.encode(batch["image"]).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))
        latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            self.device,
            self.transformer.dtype,
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(batch["text_one"], batch["text_two"])

        ## encode condition image
        control_img = self.vae.encode(batch["condition_img"]).latent_dist.sample()
        control_img = (control_img - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        control_img = FluxControlNetPipeline._pack_latents(
            control_img,
            control_img.shape[0] ,
            self.transformer.config.in_channels // 4,
            control_img.shape[2],
            control_img.shape[3],
        )

        noise = torch.randn_like(latents, device=self.device)
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme='none', batch_size=num_batches, logit_mean=0.0,
            logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise


        packed_noisy_model_input = FluxControlNetPipeline._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([1], device=self.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy_model_input,
            controlnet_cond=control_img,
            #controlnet_mode=None,
            conditioning_scale=1.0,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            )[0]
        model_pred = FluxControlNetPipeline._unpack_latents(
            model_pred,
            height=int(latents.shape[2] * vae_scale_factor / 2),
            width=int(latents.shape[3] * vae_scale_factor / 2),
            vae_scale_factor=vae_scale_factor,
        )

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigmas)

        # flow matching loss
        target = noise - latents

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

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

        pooled_prompt_embeds = self.text_encoder_one(text_one, output_hidden_states=False).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.view(pooled_prompt_embeds.shape[0], -1)
        prompt_embeds = self.text_encoder_two(text_two)[0]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=self.device, dtype=self.text_encoder_one.dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.controlnet.save_pretrained(f'{self.trainer.default_root_dir}/controlnet_weight/step_{self.global_step}')
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("controlnet."):
                del checkpoint['state_dict'][k]


class FluxControlnetLoRAModule(FluxControlnetModule):
    def __init__(self, 
                 base_model: str = "black-forest-labs/FLUX.1-dev", 
                 controlnet_model: str | None = None,
                 gradient_checkpointing: bool = False,
                 lora_rank: int = 8):
        super(FluxControlnetModule, self).__init__()
        self.tokenizer_one = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=None)
        self.tokenizer_two = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2", revision=None)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                              subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.text_encoder_two = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                                            subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae", torch_dtype=torch.bfloat16)
        self.transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="transformer", torch_dtype=torch.bfloat16)
        
        if controlnet_model is not None:
            self.controlnet = FluxControlNetModel.from_pretrained(controlnet_model)
        else:
            self.controlnet = FluxControlNetModel.from_transformer(self.transformer,
                                                                   num_layers=5,
                                                                   num_single_layers=0,
                                                                   num_attention_heads=24,
                                                                   attention_head_dim=128)
        self.controlnet.to(dtype=torch.bfloat16)

        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(False)

        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.transformer.enable_gradient_checkpointing()
            self.text_encoder_one.gradient_checkpointing_enable()
            self.text_encoder_two.gradient_checkpointing_enable()

        lora_config = LoraConfig(r=lora_rank,
                                 lora_alpha=lora_rank,
                                 init_lora_weights="gaussian",
                                 target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        #self.transformer.add_adapter(lora_config)
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()
        self.train()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.transformer.save_pretrained(f'{self.trainer.default_root_dir}/transformer_weight/step_{self.global_step}')
        # save only unet parameters
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("transformer."):
                del checkpoint['state_dict'][k]

from lightning_diffusion.model.controlnet.archs.controlnet_flux_custom import FluxControlNetMultiInputModel
from lightning_diffusion.model.controlnet.pipelines.pipeline_flux_controlnet_custom import FluxControlNetMultiInputPipeline
class FluxControlnetMultiInputModule(FluxControlnetModule):
    def __init__(self, 
                 base_model: str = "black-forest-labs/FLUX.1-dev", 
                 controlnet_model: str | None = None,
                 gradient_checkpointing: bool = False,
                 learning_rate: float = 1e-4):
        super(FluxControlnetModule, self).__init__()
        self.learning_rate = learning_rate
        self.tokenizer_one = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=None)
        self.tokenizer_two = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2", revision=None)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder_one = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                              subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.text_encoder_two = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                                            subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae", torch_dtype=torch.bfloat16)
        self.transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="transformer", torch_dtype=torch.bfloat16)
        
        if controlnet_model is not None:
            ### TODO: add colordot controlnet
            self.controlnet = FluxControlNetMultiInputModel.from_pretrained(controlnet_model, device_map=None, low_cpu_mem_usage=False)
        else:
            self.controlnet = FluxControlNetMultiInputModel.from_transformer(self.transformer,
                                                                   num_layers=5,
                                                                   num_single_layers=0,
                                                                   num_attention_heads=24,
                                                                   attention_head_dim=128)
        self.controlnet.to(dtype=torch.bfloat16)

        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.transformer.enable_gradient_checkpointing()
            self.text_encoder_one.gradient_checkpointing_enable()
            self.text_encoder_two.gradient_checkpointing_enable()
        self.train()

    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              condition_image: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int | None = 512,
              width: int | None = 512,
              num_inference_steps: int = 20,
              color_mode: str = "median"
              ) -> list[np.ndarray]:
        ### TODO: change to custom pipeline
        pipeline = FluxControlNetMultiInputPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
            controlnet=self.controlnet,
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img

            from lightning_diffusion.data.transforms import draw_random_colored_points
            color_img = load_image(img.replace("sketch", "images"))
            pil_img = pil_img.resize((height,width))
            color_img = color_img.resize((height,width))
            color_img = draw_random_colored_points(pil_img, color_img, num_range=(8,8), radius_range=(16,40), color_mode=color_mode)

            pil_img = pil_img.convert("RGB")
            image = pipeline(
                p,
                control_image=pil_img,
                control_image_2=color_img,
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

        latents = self.vae.encode(batch["image"]).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))
        latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            self.device,
            self.transformer.dtype,
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(batch["text_one"], batch["text_two"])

        ## encode condition image
        control_img = self.vae.encode(batch["condition_img"]).latent_dist.sample()
        control_img = (control_img - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        control_img = FluxControlNetPipeline._pack_latents(
            control_img,
            control_img.shape[0] ,
            self.transformer.config.in_channels // 4,
            control_img.shape[2],
            control_img.shape[3],
        )
        ## encode condition colordot image
        control_colordot_img = self.vae.encode(batch["condition_colordot_img"]).latent_dist.sample()
        control_colordot_img = (control_colordot_img - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        control_colordot_img = FluxControlNetPipeline._pack_latents(
            control_colordot_img,
            control_colordot_img.shape[0] ,
            self.transformer.config.in_channels // 4,
            control_colordot_img.shape[2],
            control_colordot_img.shape[3],
        )
        noise = torch.randn_like(latents, device=self.device)
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme='none', batch_size=num_batches, logit_mean=0.0,
            logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise


        packed_noisy_model_input = FluxControlNetPipeline._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([1], device=self.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy_model_input,
            controlnet_cond=control_img,
            controlnet_cond_2=control_colordot_img,
            #controlnet_mode=None,
            conditioning_scale=1.0,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            )[0]
        model_pred = FluxControlNetPipeline._unpack_latents(
            model_pred,
            height=int(latents.shape[2] * vae_scale_factor / 2),
            width=int(latents.shape[3] * vae_scale_factor / 2),
            vae_scale_factor=vae_scale_factor,
        )

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='none', sigmas=sigmas)

        # flow matching loss
        target = noise - latents

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        self.log("train_loss", loss)
        return loss