import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss.inverse_focal_loss import InverseFocalLoss
from ...layers.gan import randomized_multilinear_weight_initialize

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class ConditionalAdversarialTrainer:

    def __init__(
            self,
            experiment,
            source_generator,
            classifier,
            target_encoder,
            domain_discriminator,
            randomized_g,
            randomized_f,
            data_loader,
            validate_data_loader,
            n_classes,
            arg):
        self.experiment = experiment
        self.classifier = classifier
        self.source_generator = source_generator
        self.domain_discriminator = domain_discriminator
        self.target_encoder = target_encoder
        self.data_loader = data_loader
        self.validate_data_loader = validate_data_loader
        self.n_classes = n_classes
        self.arg = arg

        self.discriminator_optim = optim.Adam(
            self.domain_discriminator.parameters(), lr=self.arg.lr)
        self.target_encoder_optim = optim.Adam(
            self.target_encoder.parameters(), lr=self.arg.lr)
        self.classifier_optim = optim.Adam(
            self.classifier.parameters(), lr=self.arg.lr)
        self.source_generator_optim = optim.Adam(
            self.source_generator.parameters(), lr=self.arg.lr)
        self.domain_discriminator_criterion = InverseFocalLoss()
        self.target_encoder_criterion = InverseFocalLoss()
        self.classifier_criterion = nn.NLLLoss()
        self.randomized_g = randomized_g
        self.randomized_f = randomized_f
        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")
        self.tsne = TSNE(n_components=2)

    def train(self, epoch):
        self.target_encoder.to(self.device)
        self.classifier.to(self.device)
        self.domain_discriminator.to(self.device)
        self.source_generator.to(self.device)
        self.randomized_g = self.randomized_g.to(self.device)
        self.randomized_f = self.randomized_f.to(self.device)

        self.target_encoder_criterion.to(self.device)
        for e in range(epoch):
            target_valid_accuracy = self.validate(e)
            for i, (target_data, _) in enumerate(self.data_loader):
                target_data = target_data.to(self.device)
                batch_size = target_data.size(0)

                self.target_encoder.train()
                self.source_generator.train()
                self.classifier.train()

                discriminator_loss = self._training_discriminator(
                    target_data, batch_size)
                classifier_loss = self._training_classifier(batch_size)
                target_encoder_loss = self._training_target_encoder(
                    target_data, batch_size)
                self.experiment.log_metric('D(G(x))', target_encoder_loss)
                self.experiment.log_metric('D(x)', discriminator_loss)

            self._visualize_features(e)
            self.experiment.log_current_epoch(e)
            if e % 5 == 0:
                target_valid_accuracy = self.validate(e)
                self.experiment.log_metric(
                    'valid_target_accuracy', target_valid_accuracy)
                print("Epoch: {0} D(x): {1} D(G(x)): {2} L(D(x)): {3} target_accuracy: {4}".format(
                    e, discriminator_loss, target_encoder_loss, classifier_loss, target_valid_accuracy))

    def validate(self, e):
        self.target_encoder.eval()
        self.classifier.eval()
        accuracy = []
        for i, (target_data, target_labels) in enumerate(
                self.validate_data_loader):
            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)

            self.target_encoder.eval()
            self.classifier.eval()

            target_features = self.target_encoder(target_data)
            target_preds = self.classifier(target_features)
            _, target_preds = torch.max(target_preds, 1)
            accuracy.append(100 *
                            (target_preds == target_labels).sum().item() /
                            target_preds.size()[0])

        return torch.tensor(accuracy).mean()

    def _randomized_multilinear_map(self, features):
        mul_features = torch.mul(
            torch.mm(self.classifier(features), self.randomized_g),
            torch.mm(features, self.randomized_f)) / np.sqrt(self.arg.f_dim)
        return mul_features

    def _training_classifier(self, batch_size):
        self.target_encoder_optim.zero_grad()
        self.discriminator_optim.zero_grad()
        self.classifier_optim.zero_grad()
        self.source_generator_optim.zero_grad()

        one_hot = torch.zeros(batch_size, self.n_classes).to(self.device)
        random_labels = torch.randint(
            0,
            self.n_classes - 1,
            (batch_size,
             )).to(
            self.device).long()
        one_hot = one_hot.scatter_(
            1, torch.unsqueeze(
                random_labels, 1).to(
                self.device), 1).detach()

        z = torch.randn(batch_size, self.arg.z_dim).to(self.device).detach()
        source_features = self.source_generator(torch.cat((z, one_hot), dim=1))
        source_preds = self.classifier(source_features)

        classifier_loss = self.classifier_criterion(
            source_preds, random_labels)

        return classifier_loss

    def _training_discriminator(self, target_data, batch_size):
        self.target_encoder_optim.zero_grad()
        self.discriminator_optim.zero_grad()
        self.classifier_optim.zero_grad()
        self.source_generator_optim.zero_grad()

        batch_size = target_data.size(0)
        one_hot = torch.zeros(batch_size, self.n_classes).to(self.device)
        random_labels = torch.randint(
            0,
            self.n_classes - 1,
            (batch_size,
             )).to(
            self.device).long()
        one_hot = one_hot.scatter_(
            1, torch.unsqueeze(
                random_labels, 1).to(
                self.device), 1).detach()

        z = torch.randn(batch_size, self.arg.z_dim).to(self.device).detach()
        source_features = self.source_generator(torch.cat((z, one_hot), dim=1))
        source_mul_features = self._randomized_multilinear_map(source_features)

        source_domain_preds = self.domain_discriminator(
            source_mul_features.detach())
        source_preds = self.classifier(source_features)

        target_features = self.target_encoder(target_data)
        target_mul_features = self._randomized_multilinear_map(target_features)
        target_domain_preds = self.domain_discriminator(
            target_mul_features.detach())

        labels = torch.cat(
            (torch.zeros(batch_size).long().to(
                self.device), torch.ones(batch_size).long().to(
                self.device)), dim=0)
        preds = torch.cat((source_domain_preds, target_domain_preds), dim=0)
        discriminator_loss = self.domain_discriminator_criterion(preds, labels)
        discriminator_loss.backward()

        self.discriminator_optim.step()
        return discriminator_loss

    def _training_target_encoder(self, target_data, batch_size):
        self.target_encoder_optim.zero_grad()
        self.discriminator_optim.zero_grad()
        self.classifier_optim.zero_grad()
        self.source_generator_optim.zero_grad()

        one_hot = torch.zeros(batch_size, self.n_classes).to(self.device)
        random_labels = torch.randint(
            0,
            self.n_classes - 1,
            (batch_size,
             )).to(
            self.device).long()
        one_hot = one_hot.scatter_(
            1, torch.unsqueeze(
                random_labels, 1).to(
                self.device), 1).detach()

        z = torch.randn(batch_size, self.arg.z_dim).to(self.device).detach()
        source_features = self.source_generator(torch.cat((z, one_hot), dim=1))
        source_mul_features = self._randomized_multilinear_map(source_features)
        source_domain_preds = self.domain_discriminator(source_mul_features)

        target_features = self.target_encoder(target_data)
        target_mul_features = self._randomized_multilinear_map(target_features)
        target_domain_preds = self.domain_discriminator(target_mul_features)
        domain_preds = torch.cat(
            (source_domain_preds, target_domain_preds), dim=0)
        domain_labels = torch.cat(
            (torch.zeros(batch_size),
             torch.ones(batch_size)),
            dim=0).long().to(
            self.device)

        adversarial_loss = - \
            self.target_encoder_criterion(domain_preds, domain_labels)

        adversarial_loss.backward()
        self.target_encoder_optim.step()
        return adversarial_loss

    def _visualize_features(self, e):
        for i, (target_data, _) in enumerate(self.validate_data_loader):
            target_data = target_data.to(self.device)
            batch_size = target_data.size()[0]
            self.target_encoder.eval()

            one_hot = torch.zeros(batch_size, self.n_classes).to(self.device)
            random_labels = torch.randint(
                0,
                self.n_classes - 1,
                (batch_size,
                 )).to(
                self.device).long()
            one_hot = one_hot.scatter_(
                1, torch.unsqueeze(
                    random_labels, 1).to(
                    self.device), 1).detach()

            z = torch.randn(
                batch_size,
                self.arg.z_dim).to(
                self.device).detach()
            source_features = self.source_generator(
                torch.cat((z, one_hot), dim=1))
            target_features = self.target_encoder(target_data)
            features = torch.cat((source_features, target_features), dim=0)
            features_2dim = self.tsne.fit_transform(
                features.detach().cpu().numpy())
            plt.figure()
            plt.scatter(features_2dim[1000:, 0],
                        features_2dim[1000:, 1], c='yellow')
            self.experiment.log_figure(figure=plt)
            break
