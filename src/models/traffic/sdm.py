import torch.nn as nn
import torch.nn.functional as F
from ...layers.gan import fc_layer

class SDMG(nn.Module):

    def __init__(self, z_dim):
        super(SDMG, self).__init__()
        self.fc1 = fc_layer(z_dim + 43, 1024)
        self.fc2 = fc_layer(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.tanh(self.dropout(self.fc4(h)))
        return h

class SDMD(nn.Module):

    def __init__(self, f_dim):
        super(SDMD, self).__init__()
        self.fc1 = fc_layer(f_dim, 2048)
        self.fc2 = fc_layer(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        return F.softmax(h, dim=1)
