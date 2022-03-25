
# equivalent of main() for fsdpTrainer
# read configs, set environment and ultimately launch training

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import functools

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from distutils.command.config import config
from pathlib import Path
import os
import pprint
import argparse
from fsdptrainer.global.config import Config

from fsdptrainer.global import global_info as hq
from global.config import Config


from fsdptrainer.logging import get_dist_logger

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--backend', type=str, default='nccl', help = 'backend comms')
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--local_rank', type=int)
    return parser

def setup(config, rank, world_size, host, port, backend='nccl', local_rank=None, seed=2022, verbose=True):
    """ 3 goals:
    1 - read the user config
    2 - setup torch dist environment
    3 - setup logging 
    """

    if isinstance(config, (str, Path)):
        config = Config.from_file(config)

    hq.load_config(config)

    hq.init_global_dist(rank, world_size, backend, host, port)

    if torch.cuda.is_available():
        hq.set_device(local_rank)
    hq.set_seed(seed)

    # main dist init
    hq.init_global_dist(rank, world_size, backend, host, port)


def prepare(model, criterion, optimizer, lr_scheduler, train_loader, test_loader ):
    """ prepare model etc"""

    config = hq.config

    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False

    model = model().to(get_current_device())

    optimizer = optimizer(model.parameters())

    #



    