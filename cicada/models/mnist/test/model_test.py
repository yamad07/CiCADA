import unittest
import torch
from cicada.models.mnist.model import MNISTModel


class TestModel(unittest.TestCase):

    def test_executable(self):
        batch_size = 10
        n_classes = 10
        mnist_size = 28
        model = MNISTModel(cuda=1, n_randomized_ml=4000)
        images = torch.FloatTensor(batch_size, 1, mnist_size, mnist_size)
        preds = model(images)
        actual = preds.size()
        self.assertEqual((batch_size, n_classes), actual)

    def test_generate_conditional_features(self):
        batch_size = 10
        z_dim = 256
        model = MNISTModel(cuda=1, n_randomized_ml=4000)
        labels = torch.ones(batch_size).long()
        features = model.generate_conditional_features(labels)
        actual = features.size()
        self.assertEqual((batch_size, z_dim), actual)

    def test_calculate_source_discriminator_loss(self):
        batch_size = 10
        z_dim = 256
        mnist_size = 28
        model = MNISTModel(cuda=1, n_randomized_ml=4000)
        images = torch.FloatTensor(batch_size, 1, mnist_size, mnist_size)
        labels = torch.ones(batch_size).long()
        actual = model.calculate_source_discriminate_loss(images, labels)
        self.assertEqual(torch.Tensor, type(actual))


if __name__ == "__main__":
    unittest.main()
