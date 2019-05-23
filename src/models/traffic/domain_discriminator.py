import torch.nn as nn
import torch.nn.functional as F
from ...layers.gan import fc_layer


class DomainDiscriminator(nn.Module):
    def __init__(self, f_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = fc_layer(f_dim, 1024)
        self.fc2 = fc_layer(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = self.fc3(h)
        return F.softmax(h, dim=1)
