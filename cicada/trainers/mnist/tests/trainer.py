from cicada.trainers.mnist.trainer import MNISTTrainer
from cicada.models.mnist.model import MNISTModel
from cicada.data.damnist import DAMNIST
import unittest
import torch
import torch.utils.data as data
from torchvision import transforms


class TestTrainer(unittest.TestCase):

    def test_executable(self):
        batch_size = 10
        n_classes = 10
        mnist_size = 28
        n_features = 2000
        n_epoch = 1
        lr = 1e-2

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
        mnist_dataset = DAMNIST(
            root='./data/',
            download=True,
            source_transform=source_transform,
            target_transform=target_transform)
        data_loader = data.DataLoader(
                mnist_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cpu")
        model = MNISTModel(
                device=device,
                n_randomized_ml=2000,
                )
        trainer = MNISTTrainer().train(
                Model=MNISTModel,
                n_features=2000,
                n_epoch=n_epoch,
                lr=lr,
                device=device,
                images_loader=data_loader,
                Optimizer=torch.optim.Adam,
                )

        model = trainer.train()
        actual = type(model)
        expect = torch.nn.Module
        self.assertEqual(expect, actual)


if __name__ == "__main__":
    unittest.main()
