
# equivalent of main() for fsdpTrainer
# read configs, set environment and ultimately launch training

from pathlib import Path
import os
import pprint
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fsdptrainer.global import global_info as gt

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

def configure(config, rank, world_size, host, port, backend='nccl', local_rank=None, seed=2022, verbose=True):

    gt.load_config(config)
    gt.init_global_dist(rank, world_size, backend, host, port)

    if torch.cuda.is_available():
        gt.set_device(local_rank)
    gt.set_seed(seed)

    
    