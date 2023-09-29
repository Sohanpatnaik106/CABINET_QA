
import os
import sys
import json
import wandb
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_dataset
from typing import Any, Callable, List, Optional, Type, Union, Dict

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import (
        CPUOffload, 
        FullyShardedDataParallel, 
        FullStateDictConfig, 
        StateDictType, 
        ShardedStateDictConfig
    )


from utils import (
        process_config, 
        set_seed, 
        create_synthetic_column,
        prepare_dataloaders, 
        prepare_models,
        LightningTrainer
    )


from pytorch_lightning.callbacks import StochasticWeightAveraging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/wiki_sql/t5.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    logger = WandbLogger(project = config.logging.project, name = config.logging.name, version = config.logging.version, mode = "disabled",
                         save_dir = config.logging.save_dir, log_model = config.logging.log_model)

    set_seed(config.seed)

    if not os.path.exists(config.logging.log_dir):
        os.makedirs(config.logging.log_dir)

    if not os.path.exists(config.logging.checkpoint_dir):
        os.makedirs(config.logging.checkpoint_dir)

    if not config.training.downstream:

        print("\n************************************")
        print("*****       PRE-TRAINING     *******")
        print("************************************")

        print("\n\nLoading Dataset")

        dataset = load_dataset(config.data.data_path)
        if config.training.training_type == "column_reasoning":
            dataset = create_synthetic_column(dataset, "train")
            dataset = create_synthetic_column(dataset, "validation")
            dataset = create_synthetic_column(dataset, "test")

        train_dataloader, validation_dataloader, test_dataloader, tokenizer = prepare_dataloaders(dataset, config)

        model = prepare_models(config)
        if config.model.quantize:
            print("\n\nModel Quantization")
            param_group = {name for name, params in model.named_parameters()}
            model = torch.quantization.quantize_dynamic(model, param_group, dtype = torch.float16)

        lightning_module = LightningTrainer(config, tokenizer, model, train_dataloader = train_dataloader, 
                                            validation_dataloader = validation_dataloader, test_dataloader = test_dataloader)

        # TODO: Add the precision and other variables required for trainer in config or argparse for a generic code
        # NOTE: Currently, just check on which lines to uncomment
        if config.training.sharded:
            fsdp = FSDPStrategy(cpu_offload = False, use_orig_params = True)
            # fsdp = CustomFSDP(cpu_offload=True, use_orig_params = True)
            trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
                                sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 4, 5, 6],
                                strategy=fsdp, accelerator="gpu", precision = 16)
        else:
            # trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
            #                     sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 4, 5, 6, 7],
            #                     precision = 16, strategy = "ddp_find_unused_parameters_true")
            
            trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
                                sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 4, 5, 6])

        print("\n\nTraining")
        trainer.fit(lightning_module, train_dataloaders = train_dataloader, val_dataloaders = validation_dataloader)

        print("\n\nTesting")
        trainer.test(dataloaders = test_dataloader)

    else:

        print("\n************************************")
        print("*****       FINE-TUNING      *******")
        print("************************************")
        
        print("\n\nLoading Dataset")

        if config.data.config_name is not None:
            dataset = load_dataset(config.data.data_path, config.data.config_name)
        else:
            dataset = load_dataset(config.data.data_path)

        train_dataloader, validation_dataloader, test_dataloader, tokenizer = prepare_dataloaders(dataset, config)
        model = prepare_models(config)

        if config.model.quantize:
            print("\n\nModel Quantization")
            param_group = {name for name, params in model.named_parameters()}
            model = torch.quantization.quantize_dynamic(model, param_group, dtype = torch.float16)

        if config.model.checkpoint is not None:
            model.load_state_dict(torch.load(config.model.checkpoint), strict = False)

        lightning_module = LightningTrainer(config, tokenizer, model, train_dataloader = train_dataloader, 
                                            validation_dataloader = validation_dataloader, test_dataloader = test_dataloader)

        # TODO: Add the precision and other variables required for trainer in config or argparse for a generic code
        # NOTE: Currently, just check on which lines to uncomment
        if config.training.sharded:
            fsdp = FSDPStrategy(cpu_offload = False, use_orig_params = True)
            # fsdp = CustomFSDP(cpu_offload=False, use_orig_params = True)
            # trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
            #                     sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 4, 5, 6, 7],
            #                     strategy=fsdp, accelerator="gpu", precision = 16)
            trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
                                sync_batchnorm = config.training.sync_batchnorm, devices = [0, 1, 2, 3],
                                strategy=fsdp, accelerator="gpu")
        else:
            # trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
            #                     sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0],
            #                     precision = 16, strategy = "ddp_find_unused_parameters_true")
            trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
                                sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 4, 5, 6, 7])

        print("\n\nTraining")

        trainer.fit(lightning_module, train_dataloaders = train_dataloader, val_dataloaders = test_dataloader)

        # NOTE: Uncomment the following lines when not using sharded training
        # While using sharded training, the model checkpoint is not saved properly
        # leading to error while executing the test function

        # print("\n\nTesting")
        # trainer.test(dataloaders = test_dataloader)