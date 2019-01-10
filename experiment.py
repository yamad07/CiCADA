from comet_ml import Experiment
from src.models.source_encoder import SourceEncoder
from src.models.target_encoder import TargetEncoder
from src.models.randomized_multilinear import RandomizedMultilinear
from src.models.classifier import Classifier
from src.models.domain_discriminator import DomainDiscriminator
from src.models.sdm import SDMG, SDMD
from src.trainers.da import DomainAdversarialTrainer
from src.trainers.mcd import MaximumClassifierDiscrepancyTrainer
from src.trainers.sdm import SourceDomainModelingTrainer
from src.trainers.cada import ConditionalAdversarialTrainer
from src.trainers.cs import ConditionalSupervisedTrainer
from src.data.damnist import DAMNIST

import torch.utils.data as data
import torch
from torchvision import transforms


print('Start Experimentation')
experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                            project_name="cicada", workspace="yamad07")

source_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
target_transform = transforms.Compose([
        transforms.Resize((14, 28)),
        transforms.Pad((0, 7, 0, 7)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
mnist_dataset = DAMNIST(root='./data/', download=True, source_transform=source_transform, target_transform=target_transform)
data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)
source_validate_data_loader = data.DataLoader(mnist_dataset, batch_size=1000, shuffle=True)

validate_mnist_dataset = DAMNIST(root='./data/', train=False, download=True, source_transform=source_transform, target_transform=target_transform)
validate_data_loader = data.DataLoader(validate_mnist_dataset, batch_size=1000, shuffle=True)

source_encoder = SourceEncoder()
target_encoder = TargetEncoder()
classifier = Classifier()
domain_discriminator = DomainDiscriminator()


# domain_adversarial_trainer = DomainAdversarialTrainer(
#         experiment=experiment,
#         source_encoder=source_encoder,
#         target_encoder=target_encoder,
#         classifier_a=Classifier(),
#         classifier_b=Classifier(),
#         domain_discriminator=domain_discriminator,
#         source_domain_discriminator=SDMD(),
#         source_generator=SDMG(),
#         data_loader=data_loader,
#         valid_data_loader=validate_data_loader
#)
# domain_adversarial_trainer.train(10, 100, 100)
# domain_adversarial_trainer.validate(validate_data_loader)
# maximum_classifier_discrepancy_trainer = MaximumClassifierDiscrepancyTrainer(
#         source_encoder=source_encoder,
#         classifier_class=Classifier,
#         data_loader=data_loader,
#         validate_data_loader=source_validate_data_loader,
#         experiment=experiment
#         )
# source_encoder, classifier_a, classifier_b = maximum_classifier_discrepancy_trainer.train(5)
randomized_g = torch.randn(10, 4000).detach()
randomized_f = torch.randn(256, 4000).detach()
conditional_supervised_trainer = ConditionalSupervisedTrainer(
        source_encoder=source_encoder,
        generator=SDMG(),
        discriminator=SDMD(),
        classifier=classifier,
        randomized_f=randomized_f,
        randomized_g=randomized_g,
        data_loader=data_loader,
        validate_data_loader=source_validate_data_loader,
        experiment=experiment,
        )
source_encoder, classifier, generator = conditional_supervised_trainer.train(30)

target_encoder.load_state_dict(source_encoder.state_dict())

conditional_adversarial_trainer = ConditionalAdversarialTrainer(
        experiment=experiment,
        classifier=classifier,
        source_generator=generator,
        target_encoder=target_encoder,
        domain_discriminator=DomainDiscriminator(),
        randomized_g=randomized_g,
        randomized_f=randomized_f,
        data_loader=data_loader,
        validate_data_loader=validate_data_loader
)
conditional_adversarial_trainer.train(500)
