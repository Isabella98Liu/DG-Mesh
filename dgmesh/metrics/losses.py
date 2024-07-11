import torch

from .utils import unmask

def _chamfer_loss_v1(output_set, output_mask, target_set, target_mask):
    sizes = (~output_mask).long().sum(dim=1).tolist()
    out = output_set.flatten(0, 1)  # [B * N, C]
    out_mask = output_mask.flatten()  # [B * N]
    tgt = target_set.flatten(0, 1)  # [B * N, C]
    tgt_mask = target_mask.flatten()  # [B * N]
    out = out[~out_mask, :]  # [M, C]
    tgt = tgt[~tgt_mask, :]  # [M, C]
    outs = out.split(sizes, 0)
    tgts = tgt.split(sizes, 0)
    cd = list()
    for o, t in zip(outs, tgts):  # [m, C]
        o_ = o.unsqueeze(1).repeat(1, t.size(0), 1)  # [m, m, C]
        t_ = t.unsqueeze(0).repeat(o.size(0), 1, 1)  # [m, m, C]
        l2 = (o_ - t_).pow(2).sum(dim=-1)  # [m, m]
        tdist = l2.min(0)[0].sum()  # min over outputs
        odist = l2.min(1)[0].sum()  # min over targets
        cd.append(odist + tdist)
    loss = sum(cd) / float(len(cd))  # batch average
    return loss


def _chamfer_loss_v2(a, a_mask, b, b_mask):
    # Memory inefficient! use v1
    x, y = a, b
    x_mask, y_mask = a_mask, b_mask

    bs, x_num_points, points_dim = x.size()
    y_num_points = y.size(1)
    xx = torch.bmm(x, x.transpose(2, 1))  # [B, N, N]
    yy = torch.bmm(y, y.transpose(2, 1))  # [B, M, M]
    zz = torch.bmm(x, y.transpose(2, 1))  # [B, N, M]
    x_diag_ind = torch.arange(0, x_num_points).to(a).long()
    y_diag_ind = torch.arange(0, y_num_points).to(a).long()
    rx = xx[:, x_diag_ind, x_diag_ind].unsqueeze(-1).expand_as(zz)  # [B, N, M]
    ry = yy[:, y_diag_ind, y_diag_ind].unsqueeze(-1).expand_as(zz.transpose(2, 1))  # [B, M, N]
    P = (rx + ry.transpose(2, 1) - 2 * zz)  # [B, N, M]

    cd = list()
    for xm, ym, p in zip(x_mask, y_mask, P):
        p = p[~xm, :][:, ~ym]
        dl = p.min(0)[0].sum()  # [m,] -> scalar
        dr = p.min(1)[0].sum()  # [n,] -> scalar
        cd.append(dl + dr)
    loss = sum(cd) / float(len(cd))
    return loss


def chamfer_loss(output_set, output_mask: torch.BoolTensor,
                 target_set, target_mask: torch.BoolTensor, accelerate=False):
    if accelerate:
        from external.metrics.StructuralLosses.nn_distance import nn_distance
        assert output_set.shape[-1] == 3
        assert ((~output_mask).sum(-1) == (~target_mask).sum(-1)).all()
        output_set = unmask(output_set, output_mask).clone().contiguous()
        target_set = unmask(target_set, target_mask).clone().contiguous()
        dl, dr = nn_distance(output_set, target_set)
        cd = dl.sum(dim=1) + dr.sum(dim=1)  # Caution: ...mean() is used for evaluation
        return cd.mean()
    else:
        return _chamfer_loss_v1(output_set, output_mask, target_set, target_mask)


def emd_loss(output_set, output_mask: torch.BoolTensor,
             target_set, target_mask: torch.BoolTensor):
    from external.metrics.StructuralLosses.match_cost import match_cost
    assert output_set.shape[-1] == 3
    assert ((~output_mask).sum(-1) == (~target_mask).sum(-1)).all()
    output_set = unmask(output_set, output_mask).clone().contiguous()
    target_set = unmask(target_set, target_mask).clone().contiguous()
    emd = match_cost(output_set, target_set)  # [B,], upper bound (see PC-GAN)
    return emd.mean()
