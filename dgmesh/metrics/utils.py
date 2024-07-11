import torch


def unmask(x: torch.tensor, x_mask: torch.BoolTensor):
    # only applies for const-sized sets
    bsize = x.shape[0]
    n_points = (~x_mask).sum(-1)[0]
    assert ((~x_mask).sum(-1) == n_points).all()
    return x[~x_mask].reshape((bsize, n_points, -1))
