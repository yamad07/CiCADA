import random
from PIL import Image

from torchvision.datasets.mnist import MNIST

class DAMNIST(MNIST):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, source_transform=None,
            target_transform=None, download=False):
        super(DAMNIST, self).__init__(root=root, train=train, download=download)
        self.source_transform = source_transform
        self.target_transform = target_transform

        if self.train:
            self.source_data = self.train_data
            self.source_labels = self.train_labels
            self.target_data = self.source_data.clone()
            random.shuffle(self.target_data)
            self.target_data_for_adv = self.target_data.clone()
            random.shuffle(self.target_data_for_adv)
        else:
            self.target_data = self.test_data
            self.target_labels = self.test_labels

    def __getitem__(self, idx):
        if self.train:
            source_img = self.source_data[idx]
            source_label = self.source_labels[idx]
            target_img = self.target_data[idx]
            target_img_for_adv = self.target_data_for_adv[idx]

            source_img = Image.fromarray(source_img.numpy(), mode='L')
            target_img = Image.fromarray(target_img.numpy(), mode='L')
            target_img_for_adv = Image.fromarray(target_img_for_adv.numpy(), mode='L')

            if self.source_transform is not None:
                source_img = self.source_transform(source_img)

            if self.target_transform is not None:
                target_img = self.target_transform(target_img)
                target_img_for_adv = self.target_transform(target_img_for_adv)

            return source_img, source_label, target_img, target_img_for_adv

        target_img = self.target_data[idx]
        target_img = Image.fromarray(target_img.numpy(), mode='L')
        target_label = self.target_labels[idx]
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return target_img, target_label
