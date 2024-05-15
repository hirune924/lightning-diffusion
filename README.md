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
### Train Stable Diffusion
```sh
python scripts/main.py fit --trainer=config/trainer.yaml --model=config/stable_diffusion_model.yaml --data=config/stable_diffusion_data.yaml
```
### Train Stable Diffusion XL
```sh
python scripts/main.py fit --trainer=config/trainer.yaml --model=config/stable_diffusion_xl_model.yaml --data=config/hf_t2i_data.yaml
```

## Acknowledgement
This repo borrows the architecture design and part of the code from [diffengine](https://github.com/okotaku/diffengine).