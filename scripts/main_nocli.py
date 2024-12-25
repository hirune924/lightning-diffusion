import os
import torch
import lightning as L
from omegaconf import OmegaConf
import importlib
from lightning_diffusion.model import *
from lightning_diffusion.data import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def get_class_from_path(class_path: str) -> type:
    """
    Dynamically imports and returns a class from a string class path
    
    Args:
        class_path: Class path in 'module.submodule.ClassName' format
    Returns:
        The corresponding class
    Raises:
        ImportError: When module is not found
        AttributeError: When class is not found
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        return getattr(importlib.import_module(module_path), class_name)
    except ValueError:
        # If no '.' in path, look in current namespace
        if class_name := globals().get(class_path):
            return class_name
        raise ImportError(f"Class '{class_path}' not found")

def main():
    # configファイルの読み込み
    config = OmegaConf.load('config/anime/cogvideox_wds.yaml')
    
    # シード設定
    if config.seed_everything is not None:
        L.seed_everything(config.seed_everything)

    # データモジュールの初期化
    data_class = get_class_from_path(config.data.class_path)
    datamodule = data_class(**config.data.init_args)

    # モデルの初期化
    model_class = get_class_from_path(config.model.class_path)
    model = model_class(**config.model.init_args)
    # ロガーの初期化
    loggers_list = []
    for logger_config in config.trainer.logger:
        logger_class = get_class_from_path(logger_config.class_path)
        
        # Resolve init_args and dict_args
        init_args = OmegaConf.to_container(logger_config.init_args, resolve=True)
        dict_kwargs = OmegaConf.to_container(logger_config.dict_kwargs, resolve=True) if 'dict_kwargs' in logger_config else {}
        
        # Merge init_args and dict_args
        logger_args = {**init_args, **dict_kwargs}
        
        logger = logger_class(**logger_args)
        loggers_list.append(logger)

    # コールバックの初期化
    callbacks_list = []
    for callback_config in config.trainer.callbacks:
        callback_class = get_class_from_path(callback_config.class_path)
        init_args = OmegaConf.to_container(callback_config.init_args, resolve=True)
        dict_kwargs = OmegaConf.to_container(callback_config.dict_kwargs, resolve=True) if 'dict_kwargs' in callback_config else {}
        callback_args = {**init_args, **dict_kwargs}
        callback = callback_class(**callback_args)
        callbacks_list.append(callback)

    # トレーナーの設定
    trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
    # 元のloggerとcallbacksの設定を削除
    trainer_config.pop('logger', None)
    trainer_config.pop('callbacks', None)
    # 新しいloggerとcallbacksの設定を追加
    trainer_config['logger'] = loggers_list
    trainer_config['callbacks'] = callbacks_list
    ### ddpの設定
    import datetime
    trainer_config['strategy'] = L.pytorch.strategies.DDPStrategy(timeout=datetime.timedelta(seconds=1800*10000))
    
    trainer = L.Trainer(**trainer_config)

    # チェックポイントからの復元
    if config.ckpt_path:
        trainer.fit(model, datamodule, ckpt_path=config.ckpt_path)
    else:
        trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()