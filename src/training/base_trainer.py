import torch
import yaml
import os
import datetime
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    """
    def __init__(self, global_config_path='configs/global.yaml'):
        if global_config_path:
            self.global_config = self._load_config(global_config_path)
        
        if hasattr(self, 'global_config'):
            self.device = torch.device(self.global_config['device'])
            self.checkpoint_dir = self.global_config['checkpoint_dir']

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_checkpoint_dir(self, stage_name):
        dir_path = os.path.join(
            self.checkpoint_dir,
            stage_name,
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass
