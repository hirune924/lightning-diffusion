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
python scripts/precompute.py -c config/text_to_image/stable_diffusion.yaml --wds_save_dir=/home/n_ohta/data_cache --repeat=10
```

### TODO
* high priority
    * pre-compute-embs
    * dreambooth
    * lcm, (dmd, pcm)
* other
    * upscale
    * stable cascade
    * HDiT
    * t2i-adapter
    * textual-inversion
    * deepfloyd
    * prompt-free diffusion
    * esd
    * aspect ratio bucketing
    * save part of checkpoint
    * tips from kohya-ss/sd-scripts
    * video, 3d, ... generation

## Acknowledgement
This repo borrows the architecture design and part of the code from [diffengine](https://github.com/okotaku/diffengine), [diffusers](https://github.com/huggingface/diffusers), [naifu](https://github.com/Mikubill/naifu), [generative-models](https://github.com/Stability-AI/generative-models), [sd-scripts](https://github.com/kohya-ss/sd-scripts).

