import torch.nn as nn


def fc_layer(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
    )


def randomized_multilinear_weight_initialize(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
