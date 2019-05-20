import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ...layers.cnn import traffic_conv_layer

class TargetEncoder(nn.Module):
    def __init__(self):
        super(TargetEncoder, self).__init__()
        self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = self.dropout(self.features(x))
        a, b, c, d = h.size()
        h = h.view(a, b * c * d)
        return h
