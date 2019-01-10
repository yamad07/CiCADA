class MaximumClassifierDiscrepancyDA:
    def train(epoch):
        # training discrepancy
        for e in range(epoch):
            self.target_encoder.train()

            for i, (_, _, target_data) in enumerate(self.data_loader):
                source_data = source_data.to(self.device)
                source_labels = source_labels.to(self.device)

                discrepancy = self._minimize_discrepancy(target_data)

            val_target_accuracy = self._validate(e)
            print('Epoch: {} L(F(C(x)): {} d(p1, p2): {}'.format(e, val_target_accuracy, discrepancy))

    def _minimize_discrepancy(self, target_data):
        self.target_encoder_optim.zero_grad()

        target_features = self.target_encoder(target_data)
        target_preds_a = self.classifier_a(source_features)
        target_preds_b = self.classifier_b(source_features)
        discrepancy = F.l1_loss(target_preds_a, target_preds_b)
        discrepancy.backward()
        self.target_encoder_optim.step()
        return discrepancy

    def _validate(self, e):
        accuracy = 0
        for i, (target_data, target_labels) in enumerate(self.validate_data_loader):
            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)

            self.target_encoder.eval()
            self.classifier_a.eval()

            target_features = self.target_encoder(target_data)
            target_preds = self.classifier_a(target_features)
            _, target_preds = torch.max(target_preds, 1)
            accuracy += 100 * (target_preds == target_labels).sum().item() / target_preds.size()[0]

        accuracy /= len(self.validate_data_loader)
        return accuracy
