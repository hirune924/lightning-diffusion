# lightning-diffusion

## Getting Started
### Help
```sh
python scripts/main.py fit --help
```
### Make config
```sh
python scripts/main.py fit --print_config
```
### Train
```sh
python scripts/main.py fit -c 'your config path'
```

### TODO
* pre-compute-embs
* pixart-alpha, pixart-sigma
* inpaint
* dreambooth
* upscale
* stable cascade
* HDiT
* t2i-adapter
* prompt-free diffusion
* esd
* aspect ratio bucketing
* save part of checkpoint
* video, 3d, ... generation

## Acknowledgement
This repo borrows the architecture design and part of the code from [diffengine](https://github.com/okotaku/diffengine), [diffusers](https://github.com/huggingface/diffusers), [naifu](https://github.com/Mikubill/naifu), [generative-models](https://github.com/Stability-AI/generative-models).
