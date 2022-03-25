import logging
from distributed_logger import Distributed_Logger

__all__ = ['get_dist_logger', 'Distributed_Logger',]

def get_dist_logger(name='fsdpTrainer'):
    return Distributed_Logger.get_instance(name=name)
