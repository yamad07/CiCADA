import torch.nn as nn
import torch.nn.functional as F
from ...layers.gan import fc_layer

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = fc_layer(512, 43)

    def forward(self, x, use_log_softmax=True):
        h = self.fc1(x)
        if use_log_softmax:
            return F.log_softmax(h, dim=1)
        else:
            return F.softmax(h, dim=1)
