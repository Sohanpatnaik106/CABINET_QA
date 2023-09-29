
import os
os.chdir("../")

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

from data import WikiTQReasoningDataset, WikiTQReasoningWithoutAnswerDataset
from torch.utils.data import DataLoader


import pandas as pd

from pytorch_lightning.callbacks import StochasticWeightAveraging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/wiki_tq_reasoning/t5.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    logger = WandbLogger(project = config.logging.project, name = config.logging.name, version = config.logging.version, # mode = "disabled",
                         save_dir = config.logging.save_dir, log_model = config.logging.log_model)

    set_seed(config.seed)

    if not os.path.exists(config.logging.log_dir):
        os.makedirs(config.logging.log_dir)

    if not os.path.exists(config.logging.checkpoint_dir):
        os.makedirs(config.logging.checkpoint_dir)


    print("\n************************************")
    print("*****       BOOTSTRAPPING     *******")
    print("************************************")

    print("\n\nLoading Dataset")

    dataset = pd.read_csv(config.data.data_path)
    # dataset = load_dataset("wikitablequestions")["train"]
    train_dataset = WikiTQReasoningDataset(dataset, config)
    # train_dataset = WikiTQReasoningWithoutAnswerDataset(dataset, config)
    tokenizer = train_dataset.tokenizer

    train_dataloader = DataLoader(train_dataset, batch_size = config.training.train_batch_size, shuffle = True, num_workers = config.system.num_workers)


    model = prepare_models(config)
    if config.model.quantize:
        print("\n\nModel Quantization")
        param_group = {name for name, params in model.named_parameters()}
        model = torch.quantization.quantize_dynamic(model, param_group, dtype = torch.float16)

    lightning_module = LightningTrainer(config, tokenizer, model, train_dataloader = train_dataloader)

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
        #                     sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3, 7],
        #                     precision = 16, strategy = "ddp_find_unused_parameters_true")
        
        trainer = pl.Trainer(max_epochs = config.training.epochs, logger = logger, accumulate_grad_batches = config.training.accumulate_grad_batches,
                            sync_batchnorm = config.training.sync_batchnorm, gradient_clip_val = config.training.gradient_clip_val, devices = [0, 1, 2, 3])

    print("\n\nTraining")
    trainer.fit(lightning_module, train_dataloaders = train_dataloader)

    