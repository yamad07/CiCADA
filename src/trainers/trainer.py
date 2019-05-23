from abc import ABCMeta, abstractmethod
from comet import Experiment
import torch


class Trainer(ABCMeta):
    @abstractmethod
    def train(
            self,
            Model: Model,
            n_features: int,
            n_epoch: int,
            lr: float,
            device: torch.device,
            ):
        pass

    def log_train_process(
            self,
            experiment: Experiment,
            ):
        pass
