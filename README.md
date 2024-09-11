# lightning-diffusion

## Getting Started
### Help
```sh
python scripts/main.py fit --help
```
### Make config
```sh
python scripts/main.py fit --model=StableDiffusionModule --data=HFDataModule --print_config
```
### Train
```sh
python scripts/main.py fit -c 'your config path'
```
### Pre-compute embs
```sh
python scripts/precompute.py -c config/text_to_image/stable_diffusion.yaml --wds_save_dir=../data_cache --repeat=10
```
## Implemented
* Stable Diffusion
    * LoRA
    * ControlNet
    * inpaint
    * ip adapter
* Stable Diffusion XL
    * LoRA
    * ControlNet
    * ip adapter
* PixArt
    * LoRA
    * ControlNet
* Flux
    * LoRA
    * ControlNet
* AnymateAnyone
    * stage1/2

### TODO
* high priority
    * aspect ratio bucketing
    * save part of checkpoint
    * dreambooth
    * lcm, (dmd, pcm)
    * pixart inpaint/img2img pipeline
* other
    * upscale
    * stable cascade
    * stable diffusion 3
    * HDiT
    * t2i-adapter
    * textual-inversion
    * deepfloyd
    * prompt-free diffusion
    * esd
    * tips from kohya-ss/sd-scripts
    * video, 3d, ... generation

## Acknowledgement
This repo borrows the architecture design and part of the code from [diffengine](https://github.com/okotaku/diffengine), [diffusers](https://github.com/huggingface/diffusers), [naifu](https://github.com/Mikubill/naifu), [generative-models](https://github.com/Stability-AI/generative-models), [sd-scripts](https://github.com/kohya-ss/sd-scripts).

