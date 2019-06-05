from torch.utils.data import DataLoader


class MNISTTrainer(Trainer):
    def __init__(self, model):
        pass

    def train(
            self,
            Model: Model,
            n_features: int,
            n_epoch: int,
            lr: float,
            device: torch.device,
            data_loader: DataLoader,
    ):
        self.model = Model(n_features=n_features)
        self.device = device
        self.data_loader = data_loader

        for epoch in range(n_epoch):
            for i, (source_data, source_labels, target_data,
                    _) in enumerate(self.data_loader):
                source_labels = source_labels.to(self.device)
                target_data = target_data.to(self.device)
                self.model.train()

                discriminator_loss = self._training_discriminator(
                    source_labels, target_data)
                classifier_loss = self._training_classifier(source_labels)
                target_encoder_loss = self._training_target_encoder(
                    target_data, source_labels)

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
