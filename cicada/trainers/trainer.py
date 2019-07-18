from comet_ml import Experiment
from cicada.models.model import Model
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer(ABCMeta):
    @abstractmethod
    def train(
            self,
            Model: Model,
            n_features: int,
            n_epoch: int,
            lr: float,
            device: torch.device,
            images_loader: DataLoader,
            Optimizer: torch.optim.Optimizer,
            ) -> nn.Module:
        pass

    @abstractmethod
    def log_train_process(
            self,
            experiment: Experiment,
    ):
        pass
