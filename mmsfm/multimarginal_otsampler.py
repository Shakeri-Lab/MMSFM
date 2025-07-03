import numpy as np
import torch
from torchcfm.optimal_transport import OTPlanSampler  # type: ignore


__all__ = ['MultiMarginalPairwiseOTPlanSampler']


class MultiMarginalPairwiseOTPlanSampler(OTPlanSampler):
    def __init__(
        self, method, reg=0.05, reg_m=1.0, normalize_cost=False,
        warn=True, device='cpu'
    ):
        super().__init__(method, reg, reg_m, normalize_cost, warn)
        self.device = device

    def sample_cond_map(self, pi, i, replace=True):
        p_i = pi.sum(axis=1).reshape((-1, 1))  ## p(xi) = \sum_xj pi(xi, xj)
        p = pi / p_i                           ## pi(xj | xi) = p(xi, xj) / p(xi)
        choices = np.zeros(i.shape[0], dtype=int)
        for idxnum, idx_i in enumerate(i):
            ## select idx_j from row idx_i of table p(xj | xi)
            choices[idxnum] = np.random.choice(pi.shape[1], p=p[idx_i], replace=replace)
        return choices

    def sample_plan(self, z, replace=True):
        ## z has shape (M, bs, d)
        ## M is number of marginals
        ## bs is batch size
        ## d is dimension of data points
        M = z.shape[0]
        bs = z.shape[1]
        d = z.shape[2]

        zhat = torch.zeros(z.shape).to(self.device)
        ## first sample x0, x1
        pi = self.get_map(z[0], z[1])
        i, j = self.sample_map(pi, bs, replace=replace)
        zhat[0] = z[0, i]
        zhat[1] = z[1, j]
        ## sequentially sample x_i given x_{i-1}
        for k in range(2, M):
            ## find OT between prev sample and batch from next marginal
            pi_k = self.get_map(zhat[k-1], z[k])
            j = self.sample_cond_map(pi_k, i, replace=replace)
            zhat[k] = z[k, j]
            i = j  ## update conditioning idx

        return zhat

