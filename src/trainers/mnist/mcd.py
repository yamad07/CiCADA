import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class MaximumClassifierDiscrepancyTrainer(object):

    def __init__(self, source_encoder, classifier_class, data_loader, validate_data_loader, experiment):
        self.source_encoder = source_encoder
        self.classifier_a = classifier_class()
        self.classifier_b = classifier_class()
        self.data_loader = data_loader
        self.validate_data_loader = validate_data_loader
        self.experiment = experiment

        self.supervised_criterion = nn.NLLLoss()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.tsne = TSNE(n_components=2)

    def train(self, epoch):

        self.source_encoder.to(self.device).train()
        self.classifier_a.to(self.device).train()
        self.classifier_b.to(self.device).train()

        self.classifier_a_optim = optim.Adam(self.classifier_a.parameters(), lr=1e-4)
        self.classifier_b_optim = optim.Adam(self.classifier_b.parameters(), lr=1e-4)
        self.source_optim = optim.Adam(self.source_encoder.parameters(), lr=1e-4)

        for e in range(epoch):
            self.source_encoder.train()
            self.classifier_a.train()
            self.classifier_b.train()
            for i, (source_data, source_labels, _) in enumerate(self.data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)

                source_accuracy = self._supervised(source_data, source_labels)
                self._maximize_discrepancy(source_data, source_labels)
                discrepancy =  self._minimum_discrepancy(source_data)

            print('Epoch: {} L(F(C(x)): {} d(p1, p2): {}'.format(e, source_accuracy, discrepancy))
            self._validate(e)

        return self.source_encoder, self.classifier_a, self.classifier_b

    def _supervised(self, source_data, source_labels):
        source_features = self.source_encoder(source_data)
        source_preds_a = self.classifier_a(source_features)
        source_preds_b = self.classifier_b(source_features)
        classifier_loss = self.supervised_criterion(torch.cat((source_preds_a, source_preds_b)), torch.cat((source_labels, source_labels)))

        # backward
        classifier_loss.backward()

        self.classifier_a_optim.step()
        self.classifier_b_optim.step()
        self.source_optim.step()
        source_accuracy = self._calc_accuracy(source_preds_a, source_labels)
        return source_accuracy

    def _maximize_discrepancy(self, source_data, source_labels):
        self.classifier_a_optim.zero_grad()
        self.classifier_b_optim.zero_grad()
        self.source_optim.zero_grad()

        source_features = self.source_encoder(source_data)
        source_preds_a = self.classifier_a(source_features)
        source_preds_b = self.classifier_b(source_features)
        discrepancy = F.l1_loss(source_preds_a, source_preds_b)
        classifier_loss = self.supervised_criterion(torch.cat((source_preds_a, source_preds_b)), torch.cat((source_labels, source_labels)))
        loss = classifier_loss - discrepancy
        loss.backward()
        self.classifier_a_optim.step()
        self.classifier_b_optim.step()
        return discrepancy

    def _minimum_discrepancy(self, source_data):
        self.classifier_a_optim.zero_grad()
        self.classifier_b_optim.zero_grad()
        self.source_optim.zero_grad()

        source_features = self.source_encoder(source_data)
        source_preds_a = self.classifier_a(source_features)
        source_preds_b = self.classifier_b(source_features)
        discrepancy = F.l1_loss(source_preds_a, source_preds_b)
        discrepancy.backward()
        self.source_optim.step()
        return discrepancy

    def _calc_accuracy(self, preds, labels):
        _, preds = torch.max(preds, 1)
        accuracy = 100 * (preds == labels).sum().item() / preds.size()[0]
        return accuracy

    def _validate(self, e):
        for i, (source_data, source_labels, _) in enumerate(self.validate_data_loader):
            self.source_encoder.eval()
            source_data = source_data.to(self.device)
            source_feautures = self.source_encoder(source_data)
            source_data_features = self.tsne.fit_transform(self.source_encoder(source_data).detach().cpu().numpy())
            plt.figure()
            plt.scatter(source_data_features[:, 0], source_data_features[:, 1])
            self.experiment.log_figure(figure=plt)
            break
