import torch
import torch.distributed as dist

# local imports
from singleton import Singleton
from config import Config



from typing import Union


from fsdptrainer.logging import get_dist_logger

g_global = 'global'

class Global_Info(metaclass=Singleton):
    """ provides world_size, global and local rank per device """
    
    def __init__(self):
        self.world_sizes = {}
        self.global_ranks = {}
        self.local_ranks = {}
        self.groups = {}
        self.ranks_in_groups={}

        self.config = None

        self.logger = get_dist_logger()

    @property
    def config(self):
        return self.config
    
    def load_config(self, config: Union[str, dict] ):
        """ loads config from dict or file"""

        if isinstance(config, str):
            self.config = Config.from_file(config)
        elif isinstance(config, dict):
            self.config = Config(config)
        else:
    
    @staticmethod
    def get_global_rank(self):
        return self.global_ranks[g_global]
    
    def add_global_rank(self, int):
        self.global_ranks[g_global] = int

    def add_local_rank(self, int):
        self.local_ranks['local']=int
    
    def get_local_rank(self):
        return self.local_ranks['local'] 
    
    def get_next_global_rank(self):
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        ranks_in_group = self.get_ranks_in_group()

        return ranks_in_group[(local_rank+1)% world_size]

    def get_prev_global_rank(self):
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        ranks_in_group = self.get_ranks_in_group()

        return ranks_in_group[(local_rank-1)% world_size]
    
    def get_group(self):
        return self.groups[g_global]
    
    def add_group(self, group: dist.ProcessGroup):
        self.groups[g_global] = group
    
    def add_ranks_in_group(self, ranks:list):
        self.ranks_in_group[g_global] = ranks
    
    def get_ranks_in_group(self):
        return self.ranks_in_group[g_global]
    
    def register_dist_env(self, local_rank, world_size, process_group, ranks_in_group, mode):
        self.add_local_rank(local_rank)
        self.add_world_size(world_size)
        self.add_group(process_group)
        self.add_ranks_in_group(ranks_in_group)

    
    def global_init(self, rank, world_size, backend, host, port):
        """ main initi for global distributed environment """

        init_style = f"tcp://{host}:{port}"
        dist.init_process_group(rank=rank, world_size = world_size, backend=backend, init_method = init_style)

        self.register_dist_env(rank, world_size, None, list(range(world_size)),g_global)
        self.add_global_rank(rank)

    def destroy(self):
        dist.destroy_process_group()
        self.groups.clear()

    def set_device(self, device_ord = None):
        global_rank = self.get_global_rank()
        if device_ord is None:
            devices_per_node = torch.cuda.device_count()
            device_ord = global_rank % devices_per_node
        
        torch.cuda.set_device(device_ord)

        self.logger.info(f"process rank {global_rank} is bound to device {device_ord}")
    
    def set_seed(self, seed):
        torch.manual_seed(seed)

        global_rank = self.get_global_rank()


global_info = Global_Info()




    









