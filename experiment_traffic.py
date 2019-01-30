from comet_ml import Experiment
from src.models.traffic.source_encoder import SourceEncoder
from src.models.traffic.target_encoder import TargetEncoder
from src.models.traffic.randomized_multilinear import RandomizedMultilinear
from src.models.traffic.classifier import Classifier
from src.models.traffic.domain_discriminator import DomainDiscriminator
from src.models.traffic.sdm import SDMG, SDMD
from src.trainers.traffic.cada import ConditionalAdversarialTrainer
from src.trainers.traffic.cs import ConditionalSupervisedTrainer
from src.data.damnist import DAMNIST

import torch.utils.data as data
import torch
from torchvision import transforms
import torchvision

N_CLASSES = 43


experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                            project_name="cicada", workspace="yamad07")

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
source_dataset = torchvision.datasets.ImageFolder(root='data/TrafficDA/SYN/images/train', transform=transform)
train_source_dataloader = data.DataLoader(source_dataset, batch_size=16, shuffle=True)
val_source_dataloader = data.DataLoader(source_dataset, batch_size=43, shuffle=True)

target_dataset = torchvision.datasets.ImageFolder(root='data/TrafficDA/GTSRB/images', transform=transform)
train_target_dataloader = data.DataLoader(target_dataset, batch_size=16, shuffle=True)
val_target_dataloader = data.DataLoader(target_dataset, batch_size=43, shuffle=True)

source_encoder = SourceEncoder()
target_encoder = TargetEncoder()
classifier = Classifier()
domain_discriminator = DomainDiscriminator()

randomized_g = torch.randn(43, 4000).detach()
randomized_f = torch.randn(512, 4000).detach()
conditional_supervised_trainer = ConditionalSupervisedTrainer(
        source_encoder=source_encoder,
        generator=SDMG(),
        discriminator=SDMD(),
        classifier=classifier,
        randomized_f=randomized_f,
        randomized_g=randomized_g,
        data_loader=train_source_dataloader,
        validate_data_loader=val_source_dataloader,
        experiment=experiment,
        n_classes=N_CLASSES
        )
source_encoder, classifier, generator = conditional_supervised_trainer.train(100)

target_encoder.load_state_dict(source_encoder.state_dict())

conditional_adversarial_trainer = ConditionalAdversarialTrainer(
        experiment=experiment,
        classifier=classifier,
        source_generator=generator,
        target_encoder=target_encoder,
        domain_discriminator=DomainDiscriminator(),
        randomized_g=randomized_g,
        randomized_f=randomized_f,
        data_loader=train_target_dataloader,
        validate_data_loader=val_target_dataloader,
        n_classes=N_CLASSES
)
conditional_adversarial_trainer.train(500)
