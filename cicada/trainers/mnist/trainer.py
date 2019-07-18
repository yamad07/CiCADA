from comet_ml import Experiment
from cicada.trainers.trainer import Trainer
from cicada.models.model import Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MNISTTrainer(metaclass=Trainer):

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
        self.model = Model()
        self.device = device
        self.images_loader = images_loader
        self.optim = Optimizer(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(n_epoch):
            for i, (source_images, source_labels, _,
                    _) in enumerate(self.images_loader):
                source_images = source_images.to(self.device)
                source_labels = source_labels.to(self.device)
                classifier_loss = self.model.calculate_classifier_loss(
                        source_images, source_labels)
                discriminator_loss = self._train_classifier(
                    source_images, source_labels)

        for epoch in range(n_epoch):
            for i, (source_images, source_labels, target_data,
                    _) in enumerate(self.images_loader):
                source_labels = source_labels.to(self.device)
                target_images = target_data.to(self.device)

                discriminator_loss = self._train_domain_discriminator(
                    source_labels, target_images)
                target_encoder_loss = self._training_target_encoder(
                    target_images, source_labels)

            self._visualize_features(e)
            target_valid_accuracy = self.validate(e)
            self.experiment.log_current_epoch(e)
            self.experiment.log_metric(
                'validate_target_accuracy',
                target_valid_accuracy)
            print("Epoch: {0} D(x):{1} D(G(x)):{2} L(D(x)):{3} A:{4}".format(
                e, discriminator_loss, target_encoder_loss,
                classifier_loss, target_valid_accuracy))

    def log_train_process(
            self,
            experiment: Experiment,
    ):
        experiment.log_current_epoch(e)
        experiment.log_metric('valid_target_accuracy', target_valid_accuracy)
        print("Epoch: {0} D(x):{1} D(G(x)):{2} L(D(x)):{3} A:{4}".format(
            e, discriminator_loss, target_encoder_loss,
            classifier_loss, target_valid_accuracy))

    def _train_classifier(self, images, labels):
        self.model.zero_grad()
        loss = self.model.calculate_classifier_loss(images, labels)

        loss.backward()
        self.optim.step()

        return loss

    def _train_domain_discriminator(self, source_labels, target_images):
        self.model.zero_grad()
        loss = self.model.calculate_domain_discriminate_loss(
                source_labels, target_images)

        loss.backward()
        self.optim.step()

        return loss

    def _train_target_encoder(self, source_labels, target_images):
        self.model.zero_grad()
        loss = self.model.calculate_domain_discriminate_loss(
                source_labels, target_images)

        adversarial_loss = - loss

        adversarial_loss.backward()
        self.optim.step()
        return adversarial_loss
