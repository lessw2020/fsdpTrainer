import logging
from pathlib import Path
from typing_extensions import Self
import fsdpTrainer



class Distributed_Logger:
    instances = {}

    @staticmethod
    def get_instance(name):
        if name in Distributed_Logger.instances:
            return Distributed_Logger.instances[name]
        else:
            new_logger = Distributed_Logger(name=name)
            return new_logger
        
    def __init__(self, name):
        if name in Distributed_Logger.instances:
            print(f"Logger {name} already in use...")
        else:
            self.name = name
            self.logger = logging.getLogger(name)
            Distributed_Logger.instances[name]=self


    def set_logging_level(self, level):
        self.logger.setLevel(getattr(logging,level))
    
    def log(self, message, ranks=None):
        self.log('info', message, ranks)    
    

        
    

