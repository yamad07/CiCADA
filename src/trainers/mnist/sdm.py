import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class SourceDomainModelingTrainer(object):

    def __init__(
            self,
            source_encoder,
            source_generator,
            source_discriminator,
            classifier_a,
            classifier_b,
            data_loader,
            validate_data_loader,
            experiment):
        self.source_encoder = source_encoder
        self.source_generator = source_generator
        self.source_discriminator = source_discriminator
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        self.data_loader = data_loader
        self.validate_data_loader = validate_data_loader
        self.experiment = experiment

        self.source_discriminator_optim = optim.Adam(
            self.source_discriminator.parameters(), lr=1e-4)
        self.source_generator_optim = optim.Adam(
            self.source_generator.parameters(), lr=1e-4)

        self.source_discriminator_criterion = nn.NLLLoss()
        self.source_generator_criterion = nn.NLLLoss()
        self.supervised_criterion = nn.NLLLoss()
        self.tsne = TSNE(n_components=2)

        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")

    def train(self, epoch):

        self.source_encoder.to(self.device).train()
        self.source_generator.to(self.device).train()
        self.source_discriminator.to(self.device).train()
        self.classifier_a.to(self.device)
        self.classifier_b.to(self.device)

        for e in range(epoch):
            for i, (source_data, source_labels,
                    target_data) in enumerate(self.data_loader):
                source_data = source_data.to(self.device)
                classifier_loss, discrepancy, discriminator_loss, generator_loss = self._train_source_modeling(
                    source_data, source_labels)
                self.experiment.log_metric('D(x)', discriminator_loss)
                self.experiment.log_metric('D(G(x))', generator_loss)

            self.experiment.log_current_epoch(e)
            print("Epoch: {} D(x): {} D(G(x)): {}, d(p1, p2): {} L(F(G(x))): {}".format(
                e, discriminator_loss, generator_loss, discrepancy, classifier_loss))
            self._validate(e)
        return self.source_generator

    def _train_source_modeling(self, source_data, source_labels):
        self.source_generator_optim.zero_grad()
        self.source_discriminator_optim.zero_grad()

        # Training D(x, y)
        batch_size = source_data.size()[0]
        source_features = self.source_encoder(source_data)
        z = torch.randn(batch_size, 100).to(self.device).detach()
        source_labels = source_labels.to(self.device)
        source_one_hot = torch.zeros((batch_size, 10)).to(self.device).detach()

        source_one_hot = source_one_hot.scatter_(
            1, torch.unsqueeze(
                source_labels, 1).to(
                self.device), 1)

        source_fake_features = self.source_generator(
            torch.cat((z, source_one_hot), dim=1))

        true_preds = self.source_discriminator(
            torch.cat((source_features.detach(), source_one_hot), dim=1))
        fake_preds = self.source_discriminator(
            torch.cat((source_fake_features.detach(), source_one_hot), dim=1))
        labels = torch.cat(
            (torch.ones(batch_size).long().to(
                self.device), torch.zeros(batch_size).long().to(
                self.device)))
        preds = torch.cat((true_preds, fake_preds))

        discriminator_loss = self.source_discriminator_criterion(preds, labels)

        discriminator_loss.backward()
        self.source_discriminator_optim.step()

        # Training G(z, y)
        self.source_generator_optim.zero_grad()
        self.source_discriminator_optim.zero_grad()

        z = torch.randn(batch_size, 100).to(self.device).detach()
        source_fake_features = self.source_generator(
            torch.cat((z, source_one_hot), dim=1))

        fake_preds = self.source_discriminator(
            torch.cat((source_fake_features, source_one_hot), dim=1))
        generator_loss = self.source_generator_criterion(
            fake_preds, torch.zeros(batch_size).long().to(self.device))

        source_preds_a = self.classifier_a(source_fake_features)
        source_preds_b = self.classifier_b(source_fake_features)
        discrepancy = F.l1_loss(source_preds_a, source_preds_b)
        classifier_loss = self.supervised_criterion(
            torch.cat(
                (source_preds_a, source_preds_b)), torch.cat(
                (source_labels, source_labels)))

        loss = 0.1 * discrepancy - generator_loss

        loss.backward()
        self.source_generator_optim.step()

        return classifier_loss, discrepancy, discriminator_loss, generator_loss

    def _validate(self, e):
        for i, (source_data, source_labels, _) in enumerate(
                self.validate_data_loader):
            batch_size = source_data.size()[0]
            self.source_generator.eval()
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
            source_fake_features = self.source_generator(
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
