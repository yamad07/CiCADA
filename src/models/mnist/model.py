from ..model import Model


class MNISTModel(Model, nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.source_encoder = SourceEncoder()
        self.target_encoder = TargetEncoder()
        self.classifier = Classifier()
        self.domain_discriminator = DomainDiscriminator()
        self.source_generator = SourceGenerator()
        self.source_discriminator = SourceDiscriminator()

    def forward(self, images: torch.FloatTensor) -> preds: torch.FloatTensor:
        target_features = self.target_encode(images)
        preds = self.classify(target_features)

        return preds

    def generate_conditional_features(self, labels: torch.LongTensor) -> features: torch.FloatTensor:
        one_hot = torch.zeros(batch_size, 10)
        one_hot = one_hot.scatter_(1, torch.unsqueeze(source_labels, 1), 1)
        z = torch.randn(batch_size, 100).to(self.device)
        return self.source_generator(torch.cat((z, one_hot), dim=1).detach())


    def calculate_source_discriminate_loss(
            self,
            images: torch.FloatTensor,
            labels: torch.LongTensor) -> loss: torch.FloatTensor:

        extract_features = self.source_encoder(images)
        generate_features = self.generate_conditional_features(labels)

        fake_preds = self.domain_discriminator(generate_feautres)
        truth_preds = self.domain_discriminator(extract_feautres)
        preds = torch.cat((fake_preds, truth_preds), dim=0)

        labels = torch.cat((torch.ones(batch_size).long(), torch.zeros(batch_size).long()))
        return F.nll_loss(preds, labels)

    def calculate_domain_discriminate_loss(
            self,
            source_images: torch.FloatTensor,
            target_images: torch.FloatTensor) -> loss: torch.FloatTensor:

        source_features = self.source_encoder(source_images)
        target_features = self.target_encoder(target_images)

        source_domain_preds = self.domain_discriminator(source_feautres)
        target_domain_preds = self.domain_discriminator(target_feautres)
        preds = torch.cat((source_domain_preds, target_domain_preds), dim=0)

        labels = torch.cat((torch.ones(batch_size).long(), torch.zeros(batch_size).long()))
        return F.nll_loss(preds, labels)


    def calculate_classifier_loss(self, images: torch.FloatTensor, labels: torch.LongTensor) -> loss: torch.FloatTensor:
        pass

