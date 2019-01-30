import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ...layers.cnn import conv_layer

class TargetEncoder(nn.Module):
    def __init__(self):
        super(TargetEncoder, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet18().children())[:-1])

    def forward(self, x):
        h = self.resnet(x)
        a, b, c, d = h.size()
        h = h.view(a, b * c * d)
        return h
