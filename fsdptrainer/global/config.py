from importlib.machinery import SourcelessFileLoader
import sys
from pathlib import Path
import inspect

class Config(dict):
    """ wrapper class for dicts to enable . access
    uses .py files for config files   """

    def __init__(self, config: dict = None):
        
        if config:
            for key,value in config.items():
                self.add_item(key, value)
    

    def add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key,value)

        
    def __missing__(self, key):
        raise KeyError(key)
    
    def __setattr__(self, name, value: Any) -> None:
        return super().__setitem__(name, value)
    
    def __getattr__(self, key):
        value = None
        value = super().__getitem__(key)
        return value
    
    @staticmethod
    def from_file(filename: str):
        if not isinstance(filename, Path):
            filepath = Path(filename).absolute()
        else:
            filepath = filename.absolute()
        
        assert filepath.exists(), f"{filename} is not present..."

        suffix = filepath.suffix
        assert suffix == '.py', " requires .py files"

        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0,(filepath))
            remove_path = True
        
        mod_name = filepath.stem
        source_file = SourcelessFileLoader(fullname=str(mod_name), path = str(filepath))

        module = source_file.load_module()

        config = Config()

        for key, value in module.__dict__.items():
            if key.startswith('__') or inspect.ismodule(value) or inspect.isclass(value):
                continue
            else:
                config.add_item(key, value)

        # clear up things
        del sys.modules[mod_name]
        if remove_path:
            sys.path.pop[0]
        
        return config

        

    

