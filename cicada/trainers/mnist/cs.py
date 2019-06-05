import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...loss.inverse_focal_loss import InverseFocalLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np


class ConditionalSupervisedTrainer(object):

    def __init__(
            self,
            source_encoder,
            generator,
            discriminator,
            randomized_f,
            randomized_g,
            classifier,
            data_loader,
            validate_data_loader,
            experiment):

        self.source_encoder = source_encoder
        self.generator = generator
        self.discriminator = discriminator
        self.randomized_g = randomized_g
        self.randomized_f = randomized_f

        self.classifier = classifier
        self.data_loader = data_loader
        self.validate_data_loader = validate_data_loader
        self.experiment = experiment

        self.source_encoder_optim = optim.Adam(
            self.source_encoder.parameters(), lr=1e-3)
        self.classifier_optim = optim.Adam(
            self.classifier.parameters(), lr=1e-3)
        self.discriminator_optim = optim.Adam(
            self.discriminator.parameters(), lr=1e-4)
        self.generator_optim = optim.Adam(self.generator.parameters(), lr=1e-4)

        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")

        self.classifier_criterion = nn.NLLLoss()
        self.generator_criterion = InverseFocalLoss()
        self.discriminator_criterion = InverseFocalLoss()
        self.tsne = TSNE(n_components=2)

    def train(self, epoch):
        self.randomized_f = self.randomized_f.to(self.device)
        self.randomized_g = self.randomized_g.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        self.generator.to(self.device)
        self.source_encoder.to(self.device)

        for e in range(epoch):
            self.classifier.train()
            self.generator.train()
            self.source_encoder.train()
            self.discriminator.train()
            for i, (source_data, source_labels, _,
                    _) in enumerate(self.data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)
                source_accuracy = self._supervised(source_data, source_labels)
                generator_loss, discriminator_loss = self._source_domain_modeling(
                    source_data, source_labels)
            self._validate(e)
            if e % 1 == 0:
                source_accuracy_of_generator = self._validate_accuracy(e)
                print(
                    'Epoch: {} L(C(F(x))): {} D(x): {} D(G(z)): {} L(C(G(z))): {}'.format(
                        e,
                        source_accuracy,
                        generator_loss,
                        discriminator_loss,
                        source_accuracy_of_generator))

        return self.source_encoder, self.classifier, self.generator

    def _supervised(self, source_data, source_labels):
        source_features = self.source_encoder(source_data)
        source_preds = self.classifier(source_features)
        classifier_loss = self.classifier_criterion(
            source_preds, source_labels)
        classifier_loss.backward()

        self.source_encoder_optim.step()
        self.classifier_optim.step()
        source_accuracy = self._calc_accuracy(source_preds, source_labels)
        return source_accuracy

    def _source_domain_modeling(self, source_data, source_labels):
        batch_size = source_data.size()[0]

        # Training D(x, y)
        self.generator_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        true_source_features = self.source_encoder(source_data)
        true_mul_features = torch.mul(
            torch.mm(self.classifier(true_source_features), self.randomized_g),
            torch.mm(true_source_features, self.randomized_f)) / np.sqrt(2000)
        true_mul_preds = self.discriminator(true_mul_features.detach())

        z = torch.randn(batch_size, 100).to(self.device).detach()
        one_hot = torch.zeros((batch_size, 10)).to(self.device)
        one_hot = one_hot.scatter_(
            1, torch.unsqueeze(
                source_labels, 1).to(
                self.device), 1).detach()

        fake_features = self.generator(torch.cat((z, one_hot), dim=1))
        fake_mul_features = torch.mul(
            torch.mm(self.classifier(fake_features), self.randomized_g),
            torch.mm(fake_features, self.randomized_f)) / np.sqrt(2000)
        fake_mul_preds = self.discriminator(fake_mul_features.detach())

        mul_preds = torch.cat((true_mul_preds, fake_mul_preds), dim=0)
        labels = torch.cat(
            (torch.ones(batch_size).long().to(
                self.device), torch.zeros(batch_size).long().to(
                self.device)))

        discriminator_loss = self.discriminator_criterion(mul_preds, labels)
        discriminator_loss.backward()
        self.discriminator_optim.step()

        # Training G(z, y)
        self.generator_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        z = torch.randn(batch_size, 100).to(self.device).detach()
        fake_features = self.generator(torch.cat((z, one_hot), dim=1))
        fake_mul_features = torch.mul(
            torch.mm(self.classifier(fake_features), self.randomized_g),
            torch.mm(fake_features, self.randomized_f)) / np.sqrt(4000)
        fake_mul_preds = self.discriminator(fake_mul_features)
        generator_loss = self.generator_criterion(
            fake_mul_preds, torch.zeros(batch_size).long().to(self.device))

        fake_preds = self.classifier(fake_features)
        classifier_loss = self.classifier_criterion(fake_preds, source_labels)
        loss = classifier_loss - generator_loss
        loss.backward()
        self.generator_optim.step()
        return discriminator_loss, generator_loss

    def _calc_accuracy(self, preds, labels):
        _, preds = torch.max(preds, 1)
        accuracy = 100 * (preds == labels).sum().item() / preds.size()[0]
        return accuracy

    def _validate_accuracy(self, e):
        accuracy = 0
        for i, (source_data, source_labels) in enumerate(
                self.validate_data_loader):
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            batch_size = source_data.size()[0]
            one_hot = torch.zeros((batch_size, 10)).to(self.device)
            one_hot = one_hot.scatter_(
                1, torch.unsqueeze(
                    source_labels, 1).to(
                    self.device), 1).detach()
            z = torch.randn(batch_size, 100).to(self.device).detach()
            fake_features = self.generator(torch.cat((z, one_hot), dim=1))
            fake_preds = self.classifier(fake_features)
            accuracy += self._calc_accuracy(fake_preds, source_labels)

        accuracy /= len(self.validate_data_loader)
        return accuracy

    def _validate(self, e):
        for i, (source_data, source_labels) in enumerate(
                self.validate_data_loader):
            batch_size = source_data.size()[0]
            self.generator.eval()
            self.source_encoder.eval()
            source_data = source_data.to(self.device)
            source_one_hot = torch.zeros(
                (batch_size, 10)).to(
                self.device).detach()
            source_one_hot = source_one_hot.scatter_(
                1, torch.unsqueeze(
                    source_labels, 1).to(
                    self.device), 1)
            z = torch.randn(batch_size, 100).to(self.device).detach()
            source_features = self.source_encoder(source_data)
            source_fake_features = self.generator(
                torch.cat((z, source_one_hot), dim=1))
            source_features_2dim = self.tsne.fit_transform(torch.cat(
                (source_features, source_fake_features), dim=0).detach().cpu().numpy())
            plt.figure()
            plt.scatter(source_features_2dim[:batch_size, 0],
                        source_features_2dim[:batch_size, 1], c='pink')
            plt.scatter(source_features_2dim[batch_size:, 0],
                        source_features_2dim[batch_size:, 1], c='yellow')
            self.experiment.log_figure(figure=plt)
            break
