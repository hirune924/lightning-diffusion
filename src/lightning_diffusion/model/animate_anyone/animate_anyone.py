import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig
from lightning_diffusion.model.animate_anyone.archs.unet_2d_condition import UNet2DConditionModel
from lightning_diffusion.model.animate_anyone.archs.unet_3d_condition import UNet3DConditionModel
from transformers import CLIPVisionModelWithProjection
from lightning_diffusion.model.animate_anyone.archs.pose_guider import PoseGuider
from lightning_diffusion.model.animate_anyone.archs.mutual_self_attention import ReferenceAttentionControl
from lightning_diffusion.model.animate_anyone.archs.animate_anyone import Net
from lightning_diffusion.model.animate_anyone.pipelines.pipeline_pose2img import Pose2ImagePipeline
from lightning_diffusion.model.animate_anyone.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from PIL import Image
from diffusers.utils import load_image
from lightning_diffusion.utils.utils import load_pytorch_model
from einops import rearrange
from decord import VideoReader
import bitsandbytes as bnb

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

class AnimateAnyonePose2ImgModule(L.LightningModule):
    def __init__(self, 
                 base_model: str = "lambdalabs/sd-image-variations-diffusers", 
                 gradient_checkpointing: bool = False,
                 ucg_rate: float = 0.1,
                 noise_offset: float = 0.05,
                 snr_gamma: float = 5.0,
                 enable_zero_snr: bool = False):
        super().__init__()
        self.noise_offset = noise_offset
        self.ucg_rate = ucg_rate
        self.snr_gamma = snr_gamma
        self.enable_zero_snr = enable_zero_snr
        sched_kwargs = dict(num_train_timesteps=1000,
                            beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="scaled_linear",
                            steps_offset=1,
                            clip_sample=False)
        if self.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse")
        self.reference_unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=base_model,
                                                                   subfolder="unet")
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path=base_model,
            motion_module_path="",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                },
            )
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=base_model, subfolder="image_encoder"
            )
        
        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
        )
        
        self.vae.requires_grad_(False)
        self.image_enc.requires_grad_(False)

        # Explictly declare training models
        self.denoising_unet.requires_grad_(True)
        #  Some top layer parames of reference_unet don't need grad
        for name, param in self.reference_unet.named_parameters():
            if "up_blocks.3" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        self.pose_guider.requires_grad_(True)

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        self.net = Net(
            self.reference_unet,
            self.denoising_unet,
            self.pose_guider,
            self.reference_control_writer,
            self.reference_control_reader,
        )

        if gradient_checkpointing:
            self.reference_unet.enable_gradient_checkpointing()
            #self.denoising_unet.enable_gradient_checkpointing()
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.net.parameters())),
                                      lr=1.0e-5, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              ref_img: list[str | Image.Image],
              pose_img: list[str | Image.Image],
              height: int | None = 512,
              width: int | None = 512,
              num_inference_steps: int = 20,
              ) -> list[np.ndarray]:
        generator = torch.Generator().manual_seed(42)
        orig_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        self.image_enc.to(dtype=torch.float32)
        pipe = Pose2ImagePipeline(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler,
        )
        pipe.set_progress_bar_config(disable=True)
        images = []
        for ref, pose in zip(ref_img, pose_img):
            ref_image_pil = load_image(ref) if isinstance(ref, str) else ref
            pose_image_pil = load_image(pose) if isinstance(pose, str) else pose
            image = pipe(
                ref_image_pil,
                pose_image_pil,
                width,
                height,
                num_inference_steps,
                3.5,
                generator=generator,
            ).images
            image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (height, width, 3)
            image = Image.fromarray((image * 255).astype(np.uint8))

            images.append(np.array(image))

        del pipe
        torch.cuda.empty_cache()
        self.vae.to(dtype=orig_dtype)
        self.image_enc.to(dtype=orig_dtype)
        return images

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["img"])
        self.reference_control_reader.clear()
        self.reference_control_writer.clear()
        
        with torch.no_grad():
            latents = self.vae.encode(batch["img"]).latent_dist.sample() * self.vae.config.scaling_factor
            latents = latents.unsqueeze(2)  # (b, c, 1, h, w)

        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0,self.scheduler.config.num_train_timesteps, (num_batches, ),
            dtype=torch.int64, device=self.device)

        if self.noise_offset > 0:
            noise += self.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device
                    )
        tgt_pose_img = batch["tgt_pose"]
        tgt_pose_img = tgt_pose_img.unsqueeze(2)  # (bs, 3, 1, 512, 512)

        clip_img = batch["clip_images"]
        ref_image = batch["ref_img"]
        uncond_fwd = np.random.rand() < self.ucg_rate
        clip_img = clip_img * (1 - uncond_fwd)

        with torch.no_grad():
            ref_image_latents = self.vae.encode(ref_image).latent_dist.sample() * self.vae.config.scaling_factor  # (bs, d, 64, 64)
            clip_image_embeds = self.image_enc(clip_img).image_embeds
            image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
        
        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.net(
            noisy_latents,
            timesteps,
            ref_image_latents,
            image_prompt_embeds,
            tgt_pose_img,
            uncond_fwd,
        )
        if self.snr_gamma == 0:
            loss = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")
        else:
            snr = compute_snr(self.scheduler, timesteps)
            if self.scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]/ snr)
            loss = F.mse_loss(model_pred.float(), gt.float(), reduction="none")
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            )
            loss = loss.mean()
        self.log("train_loss", loss)
        return loss
        

class AnimateAnyonePose2VidModule(L.LightningModule):
    def __init__(self, 
                 stage1_ckpt: str,
                 ucg_rate: float = 0.1,
                 noise_offset: float = 0.05,
                 snr_gamma: float = 5.0,
                 enable_zero_snr: bool = True):
        super().__init__()
        self.noise_offset = noise_offset
        self.ucg_rate = ucg_rate
        self.snr_gamma = snr_gamma
        self.enable_zero_snr = enable_zero_snr
        sched_kwargs = dict(num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    #beta_schedule="scaled_linear",
                    beta_schedule="linear",
                    steps_offset=1,
                    clip_sample=False)
        if self.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse")
        self.reference_unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                                                                   subfolder="unet")
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path="runwayml/stable-diffusion-v1-5",
            motion_module_path="patrolli/AnimateAnyone",
            subfolder="unet",
            unet_additional_kwargs={
                    "use_inflated_groupnorm": True,
                    "unet_use_cross_frame_attention": False,
                    "unet_use_temporal_attention": False,
                    "use_motion_module": True,
                    "motion_module_resolutions": [1, 2, 4, 8],
                    "motion_module_mid_block": True,
                    "motion_module_decoder_only": False,
                    "motion_module_type": "Vanilla",
                    "motion_module_kwargs": {
                        "num_attention_heads": 8,
                        "num_transformer_block": 1,
                        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                        "temporal_position_encoding": True,
                        "temporal_position_encoding_max_len": 32,
                        "temporal_attention_dim_div": 1
                    }
                },
            )
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers", 
            subfolder="image_encoder"
            )
        
        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 64, 128)
        )

        self.denoising_unet = load_pytorch_model(stage1_ckpt, self.denoising_unet, ignore_suffix="denoising_unet")
        self.reference_unet = load_pytorch_model(stage1_ckpt, self.reference_unet, ignore_suffix="reference_unet")
        self.pose_guider = load_pytorch_model(stage1_ckpt, self.pose_guider, ignore_suffix="pose_guider")


        
        self.vae.requires_grad_(False)
        self.image_enc.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.pose_guider.requires_grad_(False)

        # Set motion module learnable
        for name, module in self.denoising_unet.named_modules():
            if "motion_modules" in name:
                for params in module.parameters():
                    params.requires_grad = True

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        self.net = Net(
            self.reference_unet,
            self.denoising_unet,
            self.pose_guider,
            self.reference_control_writer,
            self.reference_control_reader,
        )

        self.train()

    def configure_optimizers(self):
        optimizer = bnb.optim.AdamW8bit(list(filter(lambda p: p.requires_grad, self.net.parameters())), lr=1.0e-5, weight_decay=1e-2)
        return optimizer

    @torch.inference_mode()
    def forward(self,
              ref_img: list[str],
              pose_vid: list[str],
              height: int | None = 512,
              width: int | None = 512,
              clip_length: int = 24,
              num_inference_steps: int = 20,
              ) -> list[np.ndarray]:
        generator = torch.Generator().manual_seed(42)
        orig_dtype = self.denoising_unet.dtype
        self.denoising_unet.to(dtype=torch.float32)
        pipe = Pose2VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler,
        )
        pipe.set_progress_bar_config(disable=True)
        results = []
        for ref, pose in zip(ref_img, pose_vid):
            ref_image_pil = load_image(ref) if isinstance(ref, str) else ref
            pose_video = VideoReader(pose)
            pose_frame = pose_video.get_batch(list(range(clip_length)))
            
            vid = pipe(
                ref_image_pil,
                [Image.fromarray(frame) for frame in pose_frame.asnumpy()],
                width,
                height,
                clip_length,
                num_inference_steps,
                3.5,
                generator=generator,
            ).videos   # (b, c, f, h, w)


            vid = vid[0].permute(1, 2, 3, 0).cpu().numpy()  # (f, h, w, c)
            vid = (vid * 255).astype(np.uint8)

            results.append(vid)

        del pipe
        torch.cuda.empty_cache()
        self.denoising_unet.to(dtype=orig_dtype)

        return results

    def training_step(self, batch, batch_idx):
        num_batches = len(batch["pixel_values_vid"])
        self.reference_control_reader.clear()
        self.reference_control_writer.clear()
        
        with torch.no_grad():
            video_length = batch["pixel_values_vid"].shape[1]
            pixel_values_vid = rearrange(
                        batch["pixel_values_vid"], "b f c h w -> (b f) c h w"
                    )
            latents = self.vae.encode(pixel_values_vid).latent_dist.sample() * self.vae.config.scaling_factor
            latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
            
        noise = torch.randn_like(latents, device=self.device)

        timesteps = torch.randint(
            0,self.scheduler.config.num_train_timesteps, (num_batches, ),
            dtype=torch.int64, device=self.device)

        if self.noise_offset > 0:
            noise += self.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device
                    )

        pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
        pixel_values_pose = pixel_values_pose.transpose(
            1, 2
        )  # (bs, c, f, H, W)

        uncond_fwd = np.random.rand() < self.ucg_rate
        ref_img = batch["pixel_values_ref_img"]
        clip_img = batch["clip_ref_img"]
        clip_img = clip_img * (1 - uncond_fwd)

        with torch.no_grad():
            ref_image_latents = self.vae.encode(ref_img).latent_dist.sample() * self.vae.config.scaling_factor  # (bs, d, 64, 64)
            clip_image_embeds = self.image_enc(clip_img).image_embeds
            clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
        
        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.net(
            noisy_latents,
            timesteps,
            ref_image_latents,
            clip_image_embeds,
            pixel_values_pose,
            uncond_fwd,
        )

        if self.snr_gamma == 0:
            loss = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")
        else:
            snr = compute_snr(self.scheduler, timesteps)
            if self.scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]/ snr)
            loss = F.mse_loss(model_pred.float(), gt.float(), reduction="none")
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            )
            loss = loss.mean()
        self.log("train_loss", loss)
        return loss