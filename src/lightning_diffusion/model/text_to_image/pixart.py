import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel, PixArtAlphaPipeline, PixArtSigmaPipeline
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig
from lightning_diffusion.model.utils.functions import approx_standard_normal_cdf, discretized_gaussian_log_likelihood

class PixArtModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 base_transformer: str = "PixArt-alpha/PixArt-XL-2-512x512",
                 train_mode: str = "transformer_lora",
                 gradient_checkpointing: bool = True,
                 ucg_rate: float = 0.0,
                 input_perturbation_gamma: float = 0.0,
                 tokenizer_max_length: int = 120,
                 noise_offset: float = 0.0,
                 use_resolution: bool = False,
                 enable_vb_loss: bool = True):
        super().__init__()
        self.input_perturbation_gamma = input_perturbation_gamma
        self.ucg_rate = ucg_rate
        self.noise_offset = noise_offset
        self.enable_vb_loss = enable_vb_loss
        self.tokenizer_max_length = tokenizer_max_length
        self.use_resolution = use_resolution
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path="stabilityai/sd-vae-ft-ema")
        self.transformer = Transformer2DModel.from_pretrained(pretrained_model_name_or_path=base_transformer,
                                                              use_additional_conditions=use_resolution,
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
        
        if self.tokenizer_max_length == 120:
            pipe_cls = PixArtAlphaPipeline
        else:
            pipe_cls = PixArtSigmaPipeline
        pipeline = pipe_cls(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.scheduler
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for i, p in enumerate(prompt):
            generator = torch.Generator(device=self.device).manual_seed(i)
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
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

        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch["attention_mask"],
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs).sample
        
        loss_dict = self.loss(model_pred, noise, latents, timesteps, noisy_latents)
        self.log_dict(loss_dict)
        return torch.sum(torch.stack(list(loss_dict.values())))
    
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
