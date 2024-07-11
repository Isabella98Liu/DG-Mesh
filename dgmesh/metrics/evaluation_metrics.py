import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm import tqdm

# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
from metrics.StructuralLosses.match_cost import match_cost
from metrics.StructuralLosses.nn_distance import nn_distance
from .utils import unmask

# Batched CD/EMD (GPU)
def cd_cuda(sample, ref):
    return nn_distance(sample, ref)

def emd_approx_cuda(sample, ref):
    B, N, C = sample.shape
    assert C == 3, "Only for 3D point clouds"
    assert N == ref.shape[1], "Not sure what would EMD do in this case"
    emd = match_cost(sample, ref)  # (B,)
    emd_norm = emd / float(N)  # (B,)
    return emd_norm


# Batched CD (CPU), borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
def cd_cpu(sample, ref):
    x, y = sample, ref
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(sample).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


# Batched CD/EMD (wrapped)
def emd_cd(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, reduced=True):
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    assert n_sample == n_ref, "REF:%d SMP:%d" % (n_ref, n_sample)
    cd_lst = list()
    emd_lst = list()
    for b_start in range(0, n_sample, batch_size):
        b_end = min(n_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]
        dl, dr = cd_cuda(sample_batch, ref_batch) if accelerated_cd else cd_cpu(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))
        emd_batch = emd_approx_cuda(sample_batch, ref_batch)
        emd_lst.append(emd_batch)
    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)
    return {'CD': cd, 'EMD': emd}


def emd_cd_masked(sample_pcs, sample_mask, ref_pcs, ref_mask, batch_size, accelerated_cd=True, reduced=True):
    sample_pcs = unmask(sample_pcs, sample_mask)
    ref_pcs = unmask(ref_pcs, ref_mask)
    return emd_cd(sample_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd, reduced=reduced)


# Pairwise CD/EMD (GPU)
def _pairwise_emd_cd(sample_pcs, ref_pcs, batch_size=32, accelerated_cd=True):
    n_sample = sample_pcs.shape[0]
    n_ref = ref_pcs.shape[0]
    ref_pcs = ref_pcs.contiguous()
    all_cd = list()
    all_emd = list()
    for sample_b_start in tqdm(range(n_sample), total=n_sample):
        sample_batch = sample_pcs[sample_b_start]
        cd_lst = list()
        emd_lst = list()
        for ref_b_start in range(0, n_ref, batch_size):
            ref_b_end = min(n_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            dl, dr = cd_cuda(sample_batch_exp, ref_batch) if accelerated_cd else cd_cpu(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
            emd_batch = emd_approx_cuda(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))
        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)
    all_cd = torch.cat(all_cd, dim=0)  # n_sample, n_ref
    all_emd = torch.cat(all_emd, dim=0)  # n_sample, n_ref
    return all_cd, all_emd


# K-NNA, adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


# MMD, COV
def lgan_mmd_cov(all_dist):
    n_sample, n_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(n_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }

# ALL
def compute_all_metrics(sample_pcs, ref_pcs, batch_size=32, accelerated_cd=True, compute_nna=False, compute_jsd=False):
    with torch.no_grad():
        results = {}
        # MMD, COV
        M_rs_cd, M_rs_emd = _pairwise_emd_cd(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)
        res_cd = lgan_mmd_cov(M_rs_cd.t())
        results.update({"%s-CD" % k: v for k, v in res_cd.items()})
        res_emd = lgan_mmd_cov(M_rs_emd.t())
        results.update({"%s-EMD" % k: v for k, v in res_emd.items()})
        # # 1-NNA
        if compute_nna:
            M_rr_cd, M_rr_emd = _pairwise_emd_cd(ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd)
            M_ss_cd, M_ss_emd = _pairwise_emd_cd(sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)
            one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
            results.update({"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k})
            one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
            results.update({"1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k})
        if compute_jsd:
            jsd = jsd_between_point_cloud_sets(sample_pcs.cpu().detach().numpy(), ref_pcs.cpu().detach().numpy())
            results.update({'JSD': jsd})
        return results


def compute_all_metrics_masked(sample_pcs, sample_mask, ref_pcs, ref_mask, batch_size=32, accelerated_cd=True):
    with torch.no_grad():
        sample_pcs = unmask(sample_pcs, sample_mask)
        ref_pcs = unmask(ref_pcs, ref_mask)
        return compute_all_metrics(sample_pcs, ref_pcs, batch_size=batch_size, accelerated_cd=accelerated_cd)


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    with torch.no_grad():
        in_unit_sphere = True
        sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
        ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
        return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


if __name__ == "__main__":
    B, N = 2, 10
    x = torch.rand(B, N, 3)
    y = torch.rand(B, N, 3)
    min_l, min_r = cd_cuda(x.cuda(), y.cuda())
    l_dist = min_l.mean().cpu().detach().item()
    r_dist = min_r.mean().cpu().detach().item()
    print(l_dist, r_dist)
