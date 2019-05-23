from abc import ABCMeta, abstractmethod
import torch

class Model(ABCMeta):
    @abstractmethod
    def forward(self, images: torch.FloatTensor) -> preds: torch.FloatTensor:
        pass

    @abstractmethod
    def generate_conditional_features(self, labels: torch.LongTensor) -> features: torch.FloatTensor:
        pass

    @abstractmethod
    def calculate_source_discriminate_loss(self, images: torch.FloatTensor) -> loss: torch.FloatTensor:
        pass

    @abstractmethod
    def calculate_domain_discriminate_loss(self, images: torch.FloatTensor) -> loss: torch.FloatTensor:
        pass

    @abstractmethod
    def calculate_classifier_loss(self, images: torch.FloatTensor, labels: torch.LongTensor) -> loss: torch.FloatTensor:
        pass
