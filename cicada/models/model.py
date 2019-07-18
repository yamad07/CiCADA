from abc import ABCMeta, abstractmethod
import torch


class Model(ABCMeta):
    """ this abstract class is singlton pattern.
    """

    def __init__(cls, *args, **kwargs):
        type.__init__(cls, *args, **kwargs)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = type.__call__(cls, *args, **kwargs)
        return cls.instance

    @abstractmethod
    def forward(self, images: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def generate_conditional_features(
            self,
            labels: torch.LongTensor) -> torch.FloatTensor:

        pass

    @abstractmethod
    def calculate_source_discriminate_loss(
            self,
            images: torch.FloatTensor) -> torch.FloatTensor:

        pass

    @abstractmethod
    def calculate_domain_discriminate_loss(
            self,
            source_labels: torch.LongTensor,
            target_images: torch.FloatTensor) -> torch.FloatTensor:

        pass

    @abstractmethod
    def calculate_classifier_loss(
            self,
            images: torch.FloatTensor,
            labels: torch.LongTensor) -> torch.FloatTensor:

        pass
