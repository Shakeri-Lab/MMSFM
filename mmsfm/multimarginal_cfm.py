import warnings
import numpy as np
import torch
from scipy.interpolate import (
    CubicSpline,
    make_interp_spline,
    PchipInterpolator
)
from torchcfm.conditional_flow_matching import (  # type: ignore
    ConditionalFlowMatcher,
    pad_t_like_x
)
from mmsfm.multimarginal_otsampler import MultiMarginalPairwiseOTPlanSampler
from torchcfm.optimal_transport import OTPlanSampler  # type: ignore


__all__ = [
    'PairwiseConditionalFlowMatcher',
    'PairwiseExactOptimalTransportConditionalFlowMatcher',
    'PairwiseSchrodingerBridgeConditionalFlowMatcher',
    'MultiMarginalConditionalFlowMatcher',
    'MultiMarginalExactOptimalTransportConditionalFlowMatcher',
    'MultiMarginalSchrodingerBridgeConditionalFlowMatcher',
]


class PairwiseConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, zt, sigma):
        super().__init__(sigma)
        self.zt = zt
        self.dt_hat_dt = 1

    def sample_xt(self, x0, x1, t, tscaled, a, b, epsilon):
        mu_t = self.compute_mu_t(x0, x1, tscaled)
        sigma_t = self.compute_sigma_t(t, a, b)  ## equiv to using tscaled and [a, b] = [0, 1]
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt, tscaled, dt_hat_dt):
        ## compute ut(x | z) = (sigma_t_prime / sigma_t) (x - mu_t) + mu_t_prime
        ## where mu_t_prime = dmu_t / dt = dmu_t / d_t_hat * d_t_hat / dt
        ## and d_t_hat / dt = 1 / (b - a) from _t_scaler()
        # return (x1 - x0) * self.dt_hat_dt
        del t, xt, tscaled
        return (x1 - x0) * dt_hat_dt

    def sample_location_and_conditional_flow(
            self, x0, x1, t=None, t_start=None, t_end=None, return_noise=False
    ):
        if t is None:
            ## no timepoints are supplied to flow matcher
            ## assume [a, b] == [0, 1]
            t = torch.rand(x0.shape[0]).type_as(x0)
            t_start = 0
            t_end = -1
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        a = self.zt[t_start]
        b = self.zt[t_end]
        dt_hat_dt = 1 / (b - a)
        tscaled = (t - a) * dt_hat_dt

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, tscaled, a, b, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt, tscaled, dt_hat_dt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class PairwiseExactOptimalTransportConditionalFlowMatcher(PairwiseConditionalFlowMatcher):
    def __init__(self, zt, sigma):
        super().__init__(zt, sigma)
        self.ot_sampler = OTPlanSampler(method='exact')

    def compute_sigma_t(self, t, a, b):
        ## args are ignored
        return self.sigma

    def sample_location_and_conditional_flow(
            self, x0, x1, t=None, t_start=None, t_end=None, return_noise=False
    ):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(
            x0, x1, t=t, t_start=t_start, t_end=t_end, return_noise=return_noise
        )


class PairwiseSchrodingerBridgeConditionalFlowMatcher(PairwiseConditionalFlowMatcher):
    def __init__(self, zt, sigma, diff_ref, ot_method='exact'):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(zt, sigma)

        if diff_ref == 'whole':
            self.use_miniflow_sigma_t = False
        elif diff_ref == 'miniflow':
            self.use_miniflow_sigma_t = True

        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t, a, b):
        tscaled = (t - a) / (b - a)
        return self.sigma * torch.sqrt(tscaled * (1 - tscaled))

    ## overrides parent class impl
    def compute_conditional_flow(self, x0, x1, t, xt, tscaled, dt_hat_dt):
        mu_t = self.compute_mu_t(x0, x1, tscaled)
        mu_t_prime = (x1 - x0) * dt_hat_dt
        tscaled = pad_t_like_x(tscaled, x0)

        if self.use_miniflow_sigma_t:  ## compute sigma_t wrt miniflows
            sigma_t_prime_over_sigma_t = (1 - 2 * tscaled) / (2 * tscaled * (1 - tscaled) + 1e-8)
            sigma_t_prime_over_sigma_t *= dt_hat_dt  ## derivative chain rule
        else:                 ## compute sigma_t wrt whole flow
            sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        return sigma_t_prime_over_sigma_t * (xt - mu_t) + mu_t_prime

    def sample_location_and_conditional_flow(
            self, x0, x1, t=None, t_start=None, t_end=None, return_noise=False
    ):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(
            x0, x1, t=t, t_start=t_start, t_end=t_end, return_noise=return_noise
        )

    def compute_lambda(self, t, ab):
        if self.use_miniflow_sigma_t:  ## compute sigma_t wrt miniflows
            a = ab[:, 0]
            b = ab[:, 1]
        else:                 ## compute sigma_t wrt whole flow
            a = 0.
            b = 1.
        sigma_t = self.compute_sigma_t(t, a, b)
        _idx = torch.isnan(sigma_t) | torch.isinf(sigma_t)
        assert not torch.any(_idx), (a[_idx], b[_idx], t[_idx], sigma_t[_idx])  # type: ignore
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class MultiMarginalConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma, spline, monotonic, t_sampler, device='cpu'):
        super().__init__(sigma)
        ## set spline method
        if spline == 'linear':
            self._compute_store_splines = self._compute_store_linear_piecewise
            self._mu_t = self._linear_mu_t
            self._mu_t_prime = self._linear_mu_t_prime
        elif spline == 'cubic':
            self._compute_store_splines = self._compute_store_cubic_splines
            self._mu_t = self._cubic_mu_t
            self._mu_t_prime = self._cubic_mu_t_prime

        ## set cubic spline type. Ignored if spline == linear
        if monotonic:
            ## cubic hermite spline
            ## only matches up to first derivative at intermediate points
            self.spline_fn = PchipInterpolator
        else:
            ## natural cubic spline
            ## matches first AND second derivatives at intermediate points
            self.spline_fn = CubicSpline

        ## set t sampling method
        if t_sampler == 'uniform':
            self._sample_t = self._sample_t_uniformly
        elif t_sampler == 'stratified':
            self._sample_t = self._sample_t_uniformly_within_intervals

        self.device = device
        ## list of CubicSplines stored because
        ## mu and mu^\prime are computed in separate locations
        self.cs_list = []
        self.batch_slopes = None
        self.zt = None
        self.zt_diff = None


    def _sample_t_uniformly(self, zhat, zt):
        ## Samples from Unif(0, 1). zt is ignored.
        t = torch.rand(zhat.shape[1]).type_as(zhat)
        return t

    def _sample_t_uniformly_within_intervals(self, zhat, zt):
        ## returns t = [t_0, t_1, ..., t_{I}] where I is the number of intervals in zt
        ## Will do this by dividing t into sub_batches t_i
        ## let q, r = divmod(batch_size, I). Then, each t_i has size q OR q + 1
        ## More specifically, r of the I sub_batches will have size q + 1
        ## The rest will have size q. Therefore, \sum len(t_1) == batch_size
        ## Each t_i will contain uniform samples from the corresponding interval, i.e. Unif(low=zt[i], high=zt[i+1])

        n_zt_intervals = zt.shape[0] - 1
        batch_size = zhat.shape[1]
        sub_batch_size, rem = divmod(batch_size, n_zt_intervals)
        sub_batch_sizes = np.full(n_zt_intervals, sub_batch_size)
        ## select random sub_batches and increase size by 1
        ## so \sum sub_batch_sizes == batch_size
        rem_idx = np.random.choice(n_zt_intervals, size=rem, replace=False)
        sub_batch_sizes[rem_idx] += 1

        ## build master batch from sub_batches
        t = np.zeros(batch_size)
        t_idx = np.arange(batch_size)
        np.random.shuffle(t_idx)
        start = 0
        for i, sub_batch_size in enumerate(sub_batch_sizes):
            end = start + sub_batch_size
            sub_batch_idxs = t_idx[start:end]
            t_i = np.random.uniform(low=zt[i], high=zt[i+1], size=sub_batch_size)
            t[sub_batch_idxs] = t_i
            start = end

        t = torch.from_numpy(t).type_as(zhat)
        return t

    def _compute_store_cubic_splines(self, z, zt):
        ## SIDE EFFECT - STORES A LIST INTO THE OBJECT'S MEMORY
        self.cs_list.clear()
        self.cs_list = [self.spline_fn(zt, z[:, b, :].cpu().detach().numpy()) \
                        for b in range(z.shape[1])]

    def _compute_store_linear_piecewise(self, z, zt):
        ## SIDE EFFECT - STORES A LIST INTO THE OBJECT'S MEMORY
        if self.batch_slopes is None:
            ## self.batch_slopes has shape (batchsize, marginals - 1, xdim)
            self.batch_slopes = np.zeros((z.shape[1], zt.shape[0] - 1, z.shape[2]))
            self.zt = zt
            self.zt_diff = np.diff(zt)
        ## (f(x_{i+1}) - f(x_i)) / (x_{i+1} - x_i)
        self.batch_slopes = np.swapaxes(np.diff(z.cpu().detach().numpy(), axis=0), 0, 1)
        self.batch_slopes /= self.zt_diff[np.newaxis, :, np.newaxis]  # type: ignore

    def _cubic_mu_t(self, z, t):
        mu = np.zeros((z.shape[1], z.shape[2]))
        for b in range(z.shape[1]):
            cs = self.cs_list[b]
            mu[b] = cs(t[b].item())
        return mu

    def _cubic_mu_t_prime(self, z, t):
        mu_prime = np.zeros((z.shape[1], z.shape[2]))
        for b in range(z.shape[1]):
            cs = self.cs_list[b]
            ## get first derivative
            mu_prime[b] = cs(t[b].item(), 1)
        return mu_prime

    def _linear_mu_t(self, z, t):
        mu = np.zeros((z.shape[1], z.shape[2]))
        t_np = t.cpu().detach().numpy()
        t_is_1 = t_np == 1.
        ## set all positions at t = 1 to be point from target marginal
        _1_idx = np.nonzero(t_is_1)[0]
        mu[_1_idx, :] = z[-1, _1_idx, :].cpu().detach().numpy()

        ## do linear interpolation for rest of points
        batch_idx = np.nonzero(~t_is_1)[0]
        t_i = [np.nonzero(self.zt <= _t)[0][-1] for _t in t_np[batch_idx]]
        m_t = self.batch_slopes[batch_idx, t_i, :]  # type: ignore
        delta = (t_np[t_i] - self.zt[t_i])[:, np.newaxis] * m_t  # type: ignore
        mu[batch_idx, :] = z[t_i, batch_idx, :].cpu().detach().numpy() + delta

        return mu

    def _linear_mu_t_prime(self, z, t):
        ## if t == 1, then mu_prime[t] = 0
        ## otherwise, mu_prime[t] = slope from x[t_i] to x[t_{i+1}] where t_i <= t < t_{i+1}
        mu_prime = np.zeros((z.shape[1], z.shape[2]))
        batch_idx = np.nonzero(t.cpu().detach().numpy() < 1)[0]
        t_i = [np.nonzero(self.zt <= _t)[0][-1] for _t in t[batch_idx].cpu().detach().numpy()]
        mu_prime[batch_idx, :] = self.batch_slopes[batch_idx, t_i, :]  # type: ignore
        return mu_prime

    def compute_mu_t(self, z, t):
        mu = self._mu_t(z, t)

        return torch.Tensor(mu).to(self.device)

    def compute_mu_t_prime(self, z, t):
        mu_prime = self._mu_t_prime(z, t)

        return torch.Tensor(mu_prime).to(self.device)

    def sample_xt(self, z, zt, t, eps):
        self._compute_store_splines(z, zt)
        mu_t = self.compute_mu_t(z, t)
        sigma_t = self.compute_sigma_t(t, zt[0], zt[-1])
        sigma_t = pad_t_like_x(sigma_t, z[0])
        return mu_t + sigma_t * eps

    def compute_conditional_flow(self, z, t, xt, a, b):
        ## Children classes must implement this!
        raise NotImplementedError()

    def sample_location_and_conditional_flow(
            self, z, zt, t=None, return_noise=False
    ):
        zhat = self.ot_sampler.sample_plan(z)
        ## zhat.shape == (M, bs, d)
        if t is None:
            t = self._sample_t(zhat, zt)
        assert len(t) == zhat.shape[1], "t has to have batch size dimension"

        eps = super().sample_noise_like(zhat[0])
        xt = self.sample_xt(zhat, zt, t, eps)  # has side effect of storing self.cs_list
        ut = self.compute_conditional_flow(zhat, t, xt, zt[0], zt[-1])  # uses stored self.cs_list
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class MultiMarginalExactOptimalTransportConditionalFlowMatcher(MultiMarginalConditionalFlowMatcher):
    def __init__(self, sigma, spline, monotonic, t_sampler, device='cpu'):
        super().__init__(sigma, spline, monotonic, t_sampler, device=device)
        self.ot_sampler = MultiMarginalPairwiseOTPlanSampler(
            method='exact', device=device
        )

    def compute_sigma_t(self, t, a, b):
        ## all args are ignored
        return self.sigma

    def compute_conditional_flow(self, z, t, xt, a, b):
        ## return \mu_t^\prime(z)
        ut = self.compute_mu_t_prime(z, t)
        return torch.Tensor(ut).to(self.device)

    def sample_location_and_conditional_flow(
            self, z, zt, t=None, return_noise=False
    ):
        return super().sample_location_and_conditional_flow(
            z, zt, t=t, return_noise=return_noise
        )


class MultiMarginalSchrodingerBridgeConditionalFlowMatcher(MultiMarginalConditionalFlowMatcher):
    def __init__(
        self, sigma, spline, monotonic, t_sampler, diff_ref,
        method='exact', device='cpu'
    ):
        super().__init__(sigma, spline, monotonic, t_sampler, device=device)
        self.ot_sampler = MultiMarginalPairwiseOTPlanSampler(
            method=method, reg=2 * self.sigma**2, device=device
        )

        if diff_ref == 'whole':
            self.use_miniflow_sigma_t = False
        elif diff_ref == 'miniflow':
            self.use_miniflow_sigma_t = True

    def compute_sigma_t(self, t, a, b):
        ## if [a, b] == [0, 1], same as no scaling
        tscaled = (t - a) / (b - a)
        assert not torch.any(torch.isnan(tscaled) | torch.isinf(tscaled)), tscaled

        return self.sigma * torch.sqrt(tscaled * (1 - tscaled))

    ## overrides parent class impl
    def compute_conditional_flow(self, z, t, xt, a, b):
        ## return \frac{\sigma_t^\prime}{\sigma_t} (x - \mu_t) + \mu_t^\prime
        mu_t = self.compute_mu_t(z, t)
        mu_t_prime = self.compute_mu_t_prime(z, t)
        t = pad_t_like_x(t, z[0])

        if self.use_miniflow_sigma_t:  ## compute sigma_t wrt miniflows
            dtscaled_dt = 1 / (b - a)
            tscaled = (t - a) * dtscaled_dt
            sigma_t_prime_over_sigma_t = (1 - 2 * tscaled) / (2 * tscaled * (1 - tscaled) + 1e-8)
            sigma_t_prime_over_sigma_t *= dtscaled_dt  ## derivative chain rule
        else:                 ## compute sigma_t wrt whole flow
            sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)

        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + mu_t_prime
        return torch.Tensor(ut).to(self.device)

    def sample_location_and_conditional_flow(self, z, zt, t=None, return_noise=False):
        return super().sample_location_and_conditional_flow(
            z, zt, t=t, return_noise=return_noise
        )

    def compute_lambda(self, t, ab):
        if self.use_miniflow_sigma_t:  ## compute sigma_t wrt miniflows
            a = ab[:, 0]
            b = ab[:, 1]
        else:                 ## compute sigma_t wrt whole flow
            a = 0.
            b = 1.
        sigma_t = self.compute_sigma_t(t, a, b)  #############
        _idx = torch.isnan(sigma_t) | torch.isinf(sigma_t)
        assert not torch.any(_idx), (a[_idx], b[_idx], t[_idx], sigma_t[_idx])  # type: ignore
        return 2 * sigma_t / (self.sigma**2 + 1e-8)

