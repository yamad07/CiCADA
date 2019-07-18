import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cicada.models import Model
from cicada.models.mnist.source_encoder import SourceEncoder
from cicada.models.mnist.target_encoder import TargetEncoder
from cicada.models.mnist.classifier import Classifier
from cicada.models.mnist.sdm import SDMG, SDMD
from cicada.models.mnist.domain_discriminator import DomainDiscriminator
from cicada.models.mnist.randomized_multilinear import RandomizedMultilinear


class MNISTModel(nn.Module, metaclass=Model):
    def __init__(self, device=None, n_randomized_ml=None):
        super(MNISTModel, self).__init__()
        self.source_encoder = SourceEncoder()
        self.target_encoder = TargetEncoder()
        self.classifier = Classifier()
        self.domain_discriminator = DomainDiscriminator()
        self.source_generator = SDMG()
        self.source_discriminator = SDMD()
        if n_randomized_ml is None:
            raise ValueError()

        self.randomized_g = torch.FloatTensor(10, n_randomized_ml)
        self.randomized_f = torch.FloatTensor(256, n_randomized_ml)
        self.n_randomized_ml = n_randomized_ml
        self.device = device

    def forward(self, images: torch.FloatTensor) -> torch.FloatTensor:

        target_features = self.target_encoder(images)
        preds = self.classifier(target_features)
        return preds

    def generate_conditional_features(
            self,
            labels: torch.LongTensor) -> torch.FloatTensor:

        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, 10)
        one_hot = one_hot.scatter_(1, torch.unsqueeze(labels, 1), 1).float()
        z = torch.randn(batch_size, 100).to(self.device)
        return self.source_generator(torch.cat((z, one_hot), dim=1).detach())

    def calculate_source_discriminate_loss(
            self,
            images: torch.FloatTensor,
            labels: torch.LongTensor) -> torch.FloatTensor:

        batch_size = images.size(0)
        extract_features = self.source_encoder(images)
        generate_features = self.generate_conditional_features(labels)

        fake_preds = self.domain_discriminator(
            self._randomized_multilinear_map(generate_features))
        truth_preds = self.domain_discriminator(
            self._randomized_multilinear_map(extract_features))
        preds = torch.cat((fake_preds, truth_preds), dim=0)

        labels = torch.cat(
            (torch.ones(batch_size).long(),
             torch.zeros(batch_size).long()))
        return F.nll_loss(preds, labels)

    def calculate_domain_discriminate_loss(
            self,
            source_labels: torch.LongTensor,
            target_images: torch.FloatTensor) -> torch.FloatTensor:

        batch_size = source_labels.size(0)
        source_features = self.generate_conditional_features(source_labels)
        target_features = self.target_encoder(target_images)

        source_domain_preds = self.domain_discriminator(
            self._randomized_multilinear_map(source_features))
        target_domain_preds = self.domain_discriminator(
            self._randomized_multilinear_map(target_features))
        preds = torch.cat((source_domain_preds, target_domain_preds), dim=0)

        labels = torch.cat(
            (torch.ones(batch_size).long(),
             torch.zeros(batch_size).long()))
        return F.nll_loss(preds, labels)

    def calculate_classifier_loss(
            self,
            images: torch.FloatTensor,
            labels: torch.LongTensor) -> torch.FloatTensor:

        features = self.source_encoder(images)
        preds = self.classifier(features)
        loss = F.nll_loss(preds, labels)
        return loss

    def _randomized_multilinear_map(
            self,
            features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        output = torch.mul(
            torch.mm(self.classifier(features), self.randomized_g),
            torch.mm(features, self.randomized_f)) / np.sqrt(self.n_randomized_ml)
        return output
