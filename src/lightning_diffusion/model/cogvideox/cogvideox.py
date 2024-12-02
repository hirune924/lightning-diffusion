from typing import Any
import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel, CogVideoXPipeline
from transformers import T5EncoderModel, AutoTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from einops import rearrange
import bitsandbytes as bnb
from lightning_diffusion.model.cogvideox.utils.utils import prepare_rotary_positional_embeddings

class CogVideoXModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "THUDM/CogVideoX-5b", 
                 gradient_checkpointing: bool = False,
                 ucg_rate: float = 0.0,
                 enable_slicing: bool = True,
                 enable_tiling: bool = True,
                 enable_model_cpu_offload: bool = True,
                 lora_rank: int = 128,
                 lora_alpha: int = 128,
                 ):
        super().__init__()
        self.ucg_rate = ucg_rate
        self.base_model = base_model
        self.enable_slicing = enable_slicing
        self.enable_tiling = enable_tiling
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="tokenizer")
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path=base_model,
                                                       subfolder="scheduler")
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                          subfolder="text_encoder")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_name_or_path=base_model,
                                                 subfolder="vae")
        # CogVideoX-2b weights are stored in float16
        # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
        load_dtype = torch.bfloat16 if "5b" in base_model.lower() else torch.float16
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                         subfolder="transformer", torch_dtype=load_dtype)
        if enable_slicing:
            self.vae.enable_slicing()
        if enable_tiling:
            self.vae.enable_tiling()
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        transformer_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.transformer.add_adapter(transformer_lora_config)
        # For DeepSpeed training
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()
        self.train()

    def configure_optimizers(self):
        #optimizer = bnb.optim.AdamW8bit(list(filter(lambda p: p.requires_grad, self.transformer.parameters())), lr=1.0e-4, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.transformer.parameters())), lr=1.0e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.trainer.max_steps, T_mult=1)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    @torch.inference_mode()
    def forward(self,
              prompt: list[str],
              height: int | None = 480,
              width: int | None = 720,
              ) -> list[np.ndarray]:
        pipeline = CogVideoXPipeline.from_pretrained(
            self.base_model,
            transformer=self.transformer,
            scheduler=self.scheduler,
            #torch_dtype=self.dtype
            torch_dtype=torch.bfloat16
        )#.to(self.device)

        if self.enable_slicing:
            pipeline.vae.enable_slicing()
        if self.enable_tiling:
            pipeline.vae.enable_tiling()
        if self.enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=True)
        results = []
        for p in prompt:
            vid = pipeline(
                p,
                guidance_scale=6,
                height=height,
                width=width,
                max_sequence_length=self.model_config.max_text_seq_length,
                output_type="pil"
                ).frames[0]
            results.append(vid)

        del pipeline
        torch.cuda.empty_cache()

        return results

    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            b = {}
            b['prompt'] = batch[0]
            b['image'] = batch[1]
            b['video'] = batch[2]
            batch = b

        num_batches = len(batch["video"])
        if self.ucg_rate > 0:
            batch["prompt"] = ["" if np.random.rand() < self.ucg_rate else t for t in batch["prompt"]]
        batch["prompt"] = self.tokenizer(
            batch["prompt"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        video = batch["video"]
        video_length = video.shape[1]
        with torch.no_grad():
            video = rearrange(video, "b f c h w -> b c f h w")
            latents = self.vae.encode(video).latent_dist.sample() * self.vae.config.scaling_factor
            latents = rearrange(latents, "b c f h w -> b f c h w").to(memory_format=torch.contiguous_format)

        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0,self.scheduler.config.num_train_timesteps, (num_batches, ),
            dtype=torch.int64, device=self.device)
        # Prepare rotary embeds
        batch_size, num_frames, num_channels, height, width = latents.shape
        VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(self.vae.config.block_out_channels) - 1)
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * VAE_SCALE_FACTOR_SPATIAL,
                width=width * VAE_SCALE_FACTOR_SPATIAL,
                num_frames=num_frames,
                vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                patch_size=self.model_config.patch_size,
                attention_head_dim=self.model_config.attention_head_dim,
                device=self.device,
            )
            if self.model_config.use_rotary_positional_embeddings
            else None
        )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["prompt"], return_dict=False)[0]

        model_pred = self.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
        model_pred = self.scheduler.get_velocity(model_pred, noisy_latents, timesteps)
        weights = 1 / (1 - self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)[timesteps])
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean(
                    (weights * (model_pred - latents) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
        self.log("train_loss", loss)
        return loss
    
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        transformer_lora_layers_to_save = get_peft_model_state_dict(self.model)
        CogVideoXPipeline.save_lora_weights(
                f'{self.trainer.default_root_dir}/motion_module_weight/step_{self.global_step}',
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
        list_keys = list(checkpoint['state_dict'].keys())
        for k in list_keys:
            if not k.startswith("transformer"):
                del checkpoint['state_dict'][k]