import torch.nn as nn


class RandomizedMultilinear(nn.Module):

    def __init__(self, n_input, n_outputs):
        super(RandomizedMultilinear, self).__init__()
        self.f = nn.Linear(n_input, n_outputs)

    def forward(self, input):
        return self.f(input)
