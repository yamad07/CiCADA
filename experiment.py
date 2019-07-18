from comet_ml import Experiment
from src.models.mnist.source_encoder import SourceEncoder
from src.models.mnist.target_encoder import TargetEncoder
from src.models.mnist.randomized_multilinear import RandomizedMultilinear
from src.models.mnist.classifier import Classifier
from src.models.mnist.domain_discriminator import DomainDiscriminator
from src.models.mnist.sdm import SDMG, SDMD
from src.trainers.mnist.da import DomainAdversarialTrainer
from src.trainers.mnist.mcd import MaximumClassifierDiscrepancyTrainer
from src.trainers.mnist.sdm import SourceDomainModelingTrainer
from src.trainers.mnist.cada import ConditionalAdversarialTrainer
from src.trainers.mnist.cs import ConditionalSupervisedTrainer
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
continuous_size = [1, 3, 5, 7]

mnist_dataset = DAMNIST(
    root='./data/',
    download=True,
    source_transform=source_transform,
    target_transform=target_transform)
data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)

validate_mnist_dataset = DAMNIST(
    root='./data/',
    train=False,
    download=True,
    source_transform=source_transform,
    target_transform=target_transform)
source_validate_data_loader = data.DataLoader(
    validate_mnist_dataset, batch_size=1000, shuffle=True)
target_validate_data_loader = data.DataLoader(
    validate_mnist_dataset, batch_size=1000, shuffle=True)

source_encoder = SourceEncoder()
target_encoder = TargetEncoder()
classifier = Classifier()
domain_discriminator = DomainDiscriminator()

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
source_encoder, classifier, generator = conditional_supervised_trainer.train(
    10)

target_encoder.load_state_dict(source_encoder.state_dict())

conditional_adversarial_trainer = ConditionalAdversarialTrainer(
    experiment=experiment,
    classifier=classifier,
    source_generator=generator,
    target_encoder=target_encoder,
    domain_discriminator=DomainDiscriminator(),
    randomized_g=randomized_g,
    randomized_f=randomized_f,
    data_loader=None,
    validate_data_loader=None
)
for size in continuous_size:
    target_transform = transforms.Compose([
        transforms.Resize((int(28 - size * 2), 28)),
        transforms.Pad((0, size, 0, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    mnist_dataset = DAMNIST(
        root='./data/',
        download=True,
        source_transform=source_transform,
        target_transform=target_transform)
    data_loader = data.DataLoader(mnist_dataset, batch_size=16, shuffle=True)

    validate_mnist_dataset = DAMNIST(
        root='./data/',
        train=False,
        download=True,
        source_transform=source_transform,
        target_transform=target_transform)
    target_validate_data_loader = data.DataLoader(
        validate_mnist_dataset, batch_size=1000, shuffle=True)

    conditional_adversarial_trainer.set_data_loader(data_loader)
    conditional_adversarial_trainer.set_validate_data_loader(
        target_validate_data_loader)
    conditional_adversarial_trainer.train(50)
