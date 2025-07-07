import numpy as np
from scipy.spatial import distance_matrix
import ot
import time
from abc import ABC, abstractmethod
import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm
from os.path import isdir

from torchcfm.conditional_flow_matching import *             # type: ignore
from torchcfm.utils import plot_trajectories, torch_wrapper  # type: ignore

from mmsfm.models import MLP

from mmsfm.multimarginal_cfm import (
    PairwiseExactOptimalTransportConditionalFlowMatcher,
    PairwiseSchrodingerBridgeConditionalFlowMatcher,
    MultiMarginalExactOptimalTransportConditionalFlowMatcher,
    MultiMarginalSchrodingerBridgeConditionalFlowMatcher
)

from utils import (
    timer_func,
)

from MIOFlow.losses import MMD_loss  # type: ignore


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size, sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    ## Drift
    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x).flatten(start_dim=1) + self.score(x).flatten(start_dim=1)

    ## Diffusion
    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


class BaseAgent(ABC):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        self.run = None

        self.zt_all = args.zt
        self.zt_rem_idxs = zt_rem_idxs
        self.zt = args.zt[zt_rem_idxs]
        self.method = args.method
        self.eval_method = args.eval_method
        self.sigma = args.sigma
        self.batch_size = args.batch_size
        self.modelname = args.modelname
        self.w_len = args.w_len
        self.modeldepth = args.modeldepth
        self.lr = args.lr
        self.n_steps = args.n_steps
        self.n_infer = args.n_infer
        self.t_infer = args.t_infer
        self.reg = args.reg
        self.dim = dim
        self.device = device
        self.use_cuda = self.device == 'cuda'

        ## Set Up FlowMatcher, Model, and Optimizer
        self.is_sb = args.flowmatcher == 'sb'
        self.diff_ref = args.diff_ref

        self.FM = self._build_FM()
        self.model = self._build_model()
        self.score_model = self._build_model() if self.is_sb else None
        self.optimizer = self._build_optimizer()

        ## Set Up batch_fn
        if args.rand_heldouts:
            self.batch_fn = self._get_batch_rand_heldout
        else:
            self.batch_fn = self._get_batch
        print(f'Using batch function: {self.batch_fn.__name__}()')

        self.step_counter = 0  ## internal counter for used for logging

    def save_models(self, outdir):
        torch.save(self.model.state_dict(), f'{outdir}/flow_model.pth')
        if self.is_sb:
            torch.save(self.score_model.state_dict(), f'{outdir}/score_model.pth')  # type: ignore

    ###### SET UP METHODS ######
    def set_run(self, run):
        self.run = run

    def _build_model(self):
        if self.modelname == 'mlp':
            model = (
                MLP(
                    dim=self.dim,
                    w=self.w_len,
                    depth=self.modeldepth,
                    time_varying=True,
                ).to(self.device)
            )
        else:
            raise ValueError('Please use a valid model type.')

        return model

    def _build_optimizer(self):
        params = list(self.model.parameters())  # type: ignore
        if self.is_sb:
            params += list(self.score_model.parameters())  # type: ignore

        return torch.optim.AdamW(params, self.lr)

    @abstractmethod
    def _build_FM(self):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @abstractmethod
    def _get_batch(self, X, return_noise=False):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @abstractmethod
    def _get_batch_rand_heldout(self, X, return_noise=False):
        raise NotImplementedError('Not Implemented for Abstract Class')
    ############################

    ######    TRAINING    ######
    @abstractmethod
    def _train(self, X, pbar=None):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @timer_func
    def train(self, X, pbar=None):
        return self._train(X, pbar=pbar)
    ############################

    ######   PRED & EVAL  ######
    @timer_func
    def traj_gen(self, x0):
        solver = 'euler' if self.is_sb else 'dopri5'

        node = NeuralODE(
            torch_wrapper(self.model),
            solver=solver,
            sensitivity='adjoint',
            atol=1e-4,
            rtol=1e-4
        )
        x0 = (
            torch.from_numpy(x0[:self.n_infer])
            .float()
            .to(self.device)
        )
        t_span = torch.linspace(0, 1, self.t_infer)

        print('Solving ODE and computing trajectories...')
        with torch.no_grad():
            ode_traj = node.trajectory(
                x0,
                t_span=t_span
            ).cpu().numpy()

        if self.is_sb:
            sde = SDE(
                self.model,
                self.score_model,
                input_size=(self.dim,),
                sigma=self.sigma
            ).to(self.device)

            print('Solving SDE and computing trajectories...')
            with torch.no_grad():
                sde_traj = torchsde.sdeint(
                    sde,
                    x0,
                    ts=t_span.to(self.device)
                ).cpu().numpy()  # type: ignore
        else:
            sde_traj = None

        return ode_traj, sde_traj

    def _get_traj_at_zt(self, traj, zt):
        ## gets closest points in inferred trajectory to true timepoints zt
        tspan = np.linspace(0, 1, traj.shape[0])

        ## get nearest zt_idx in tspan
        if self.eval_method == 'nearest':
            zt_idx = np.zeros(zt.shape[0], dtype=int)

            for i, t in enumerate(zt):
                dist_from_t = np.abs(tspan - t)
                zt_idx[i] = np.argmin(dist_from_t)

            traj_at_zt = traj[zt_idx, :, :]

        ## interpolate
        elif self.eval_method == 'interp':
            dt = np.diff(tspan)[0]
            traj_at_zt = np.zeros((zt.shape[0], traj.shape[1], traj.shape[2]))
            for i, t in enumerate(zt[:-1]):
                j, rem = np.divmod(t, dt)
                j = int(j)
                traj_at_zt[i] = (traj[j+1] * rem) + ((1 - rem) * traj[j])
            traj_at_zt[-1] = traj[-1]

        else:
            raise ValueError(f'Argument method = {self.eval_method} is not defined.')

        return traj_at_zt

    @timer_func
    def traj_eval(self, testdata, traj, zt=None):
        if zt is not None:
            ## evaluate at given zt
            traj_at_zt = self._get_traj_at_zt(traj, zt)
            m = zt.shape[0]
        else:
            ## assume testdata and traj both have same number of timepoints
            traj_at_zt = traj
            m = traj.shape[0]

        W_euclid = np.zeros(m)
        W_euclid_reg = np.zeros(m)
        W_sqeuclid = np.zeros(m)
        W_sqeuclid_reg = np.zeros(m)
        MMD_G = np.zeros(m)
        MMD_M = np.zeros(m)

        print('Computing Evaluation Losses...')
        for i in tqdm(range(m)):
            u_i = testdata[i]
            v_i = traj_at_zt[i]

            M = ot.dist(u_i, v_i, metric='euclidean')
            M2 = ot.dist(u_i, v_i, metric='sqeuclidean')
            a = ot.unif(u_i.shape[0])
            b = ot.unif(v_i.shape[0])

            ## use sinkhorn_log for entropic regularized OT
            ## to avoid numerical instability at the cost of speed
            ## because regular 'sinkhorn_knopp' uses K = np.exp(M / (-reg))
            ## which can often cause nans and break the calculations
            W_euclid[i] = ot.emd2(a, b, M)
            W_euclid_reg[i] = ot.sinkhorn2(a, b, M, reg=self.reg, method='sinkhorn_log')
            W_sqeuclid[i] = ot.emd2(a, b, M2)
            W_sqeuclid_reg[i] = ot.sinkhorn2(a, b, M2, reg=self.reg, method='sinkhorn_log')
            MMD_G[i] = self.mmd_gaussian(u_i, v_i)
            MMD_M[i] = self.mmd_mean(u_i, v_i)

        eval_losses_dict = {
            'W_euclid': W_euclid,
            'W_euclid_reg': W_euclid_reg,
            'W_sqeuclid': W_sqeuclid,
            'W_sqeuclid_reg': W_sqeuclid_reg,
            'MMD_G': MMD_G,
            'MMD_M': MMD_M
        }

        return eval_losses_dict

    def mmd_gaussian(self, u, v):
        ## wrapper for MMD_loss from MIOFlow
        mmd_fn = MMD_loss()
        k = np.min([u.shape[0], v.shape[0], self.n_infer])
        u_idx = np.random.choice(
            np.arange(u.shape[0], dtype=int), size=k, replace=False
        )
        v_idx = np.random.choice(
            np.arange(v.shape[0], dtype=int), size=k, replace=False
        )
        uu = torch.from_numpy(u[u_idx])
        vv = torch.from_numpy(v[v_idx])
        return mmd_fn(uu, vv).item()

    def mmd_mean(self, u, v):
        mu_u = np.mean(u, axis=0)
        mu_v = np.mean(v, axis=0)
        delta = mu_u - mu_v
        return np.einsum('i,i->', delta, delta)
    ############################


class PairwiseAgent(BaseAgent):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        super().__init__(args, zt_rem_idxs, dim, device=device)

    ###### SET UP METHODS ######
    def _build_FM(self):
        if not self.is_sb:    ## OT
            FM = PairwiseExactOptimalTransportConditionalFlowMatcher(
                zt=self.zt, sigma=self.sigma
            )
        else:                 ## SB
            FM = PairwiseSchrodingerBridgeConditionalFlowMatcher(
                zt=self.zt, sigma=self.sigma,
                diff_ref=self.diff_ref, ot_method=self.method
            )

        return FM

    def _get_batch(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        ztdiff = np.diff(self.zt)
        n_times = len(self.zt)
        for t_idx in range(n_times - 1):
            x0 = (
                torch.from_numpy(
                    X[t_idx][np.random.randint(X[t_idx].shape[0], size=self.batch_size)]
                )
                .float()
                .to(self.device)
            )
            x1 = (
                torch.from_numpy(
                    X[t_idx+1][np.random.randint(X[t_idx+1].shape[0], size=self.batch_size)]
                )
                .float()
                .to(self.device)
            )
            _t = torch.rand(x0.shape[0]).type_as(x0)
            _t = self.zt[t_idx] + (_t * ztdiff[t_idx])  ## _t spans [zt[t_idx], zt[t_idx+1]]
            if return_noise:
                t, xt, ut, eps = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_idx, t_end=t_idx+1, return_noise=return_noise
                )
                noises.append(eps)  # type: ignore
            else:
                t, xt, ut = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_idx, t_end=t_idx+1, return_noise=return_noise
                )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)

        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 1):
                k1 = k + 1
                curr_ab = torch.Tensor(self.zt[[k, k1]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab

    def _get_batch_rand_heldout(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = []
        ztdiff = np.diff(self.zt)
        n_times = len(X)
        t_heldout = np.random.randint(1, n_times - 1)  # only hold out intermediate timepoints
        rem_margs = np.delete(np.arange(len(X), dtype=int), t_heldout)  # list of timepoints after holding out
        for i in range(len(X) - 2):
            t_start = rem_margs[i]
            t_end = rem_margs[i+1]
            x0 = (
                torch.from_numpy(
                    X[t_start][np.random.randint(X[t_start].shape[0], size=self.batch_size)])
                .float()
                .to(self.device)
            )
            x1 = (
                torch.from_numpy(
                    X[t_end][np.random.randint(X[t_end].shape[0], size=self.batch_size)]
                )
                .float()
                .to(self.device)
            )
            _t = torch.rand(x0.shape[0]).type_as(x0)
            _t = self.zt[t_start] + (_t * (self.zt[t_end] - self.zt[t_start]))
            if return_noise:
                t, xt, ut, eps = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_start, t_end=t_end, return_noise=return_noise
                )
                noises.append(eps)
            else:
                t, xt, ut = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_start, t_end=t_end, return_noise=return_noise
                )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)

        if return_noise:
            noises = torch.cat(noises)
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 1):
                k1 = k + 1
                curr_ab = torch.Tensor(self.zt[[k, k1]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab
    ############################

    ######    TRAINING    ######
    def _train(self, X, pbar=None):
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        print('Training...')

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            ## eps is None if OT, float if SB
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            xt_t = torch.cat([xt, t[:, None]], dim=-1)
            vt = self.model(xt_t)
            loss = torch.mean((vt - ut) ** 2)
            flow_losses[i] = loss.item()

            if self.is_sb:
                lambda_t = self.FM.compute_lambda(t, ab)
                st = self.score_model(xt_t)  # type: ignore
                score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                score_losses[i] = score_loss.item()  # type: ignore
                loss += score_loss

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore
                    commit=False
                )

            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'n_steps'   : self.step_counter
                }
            )  ## commit loss to wandb servers

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################


class TripletAgent(BaseAgent):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        self.t_sampler = args.t_sampler
        self.spline = args.spline
        self.monotonic = args.monotonic
        super().__init__(args, zt_rem_idxs, dim, device=device)

    ###### SET UP METHODS ######
    def _build_FM(self):
        if not self.is_sb:    ## OT
            FM = MultiMarginalExactOptimalTransportConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, device=self.device
            )
        else:            ## SB
            FM = MultiMarginalSchrodingerBridgeConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, diff_ref=self.diff_ref,
                method=self.method, device=self.device
            )

        return FM

    def _get_batch(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        for k in range(len(X) - 2):
            z = torch.zeros(3, self.batch_size, self.dim)
            for z_idx, i in enumerate(range(k, k+3)):
                z[z_idx] = torch.from_numpy(
                    X[i][np.random.randint(X[i].shape[0], size=self.batch_size)]
                ).float()
            z = z.to(self.device)

            t, xt, ut, *eps = self.FM.sample_location_and_conditional_flow(
                z, self.zt[k:k+3], return_noise=return_noise
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps[0])  # type: ignore

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 2):
                k1 = k + 1
                k2 = k + 2
                curr_ab = torch.Tensor(self.zt[[k, k2]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab

    def _get_batch_rand_heldout(self, X, return_noise=False):
        t_heldout = np.random.randint(1, len(X) - 1)  # heldout idx
        rem_margs = np.delete(np.arange(len(X), dtype=int), t_heldout)  # list of remaining idxs
        rem_zts = self.zt[rem_margs]

        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        for k in range(len(X) - 3):
            z = torch.zeros(3, self.batch_size, self.dim)
            for z_idx, i in enumerate(range(k, k+3)):
                m = rem_margs[i]
                z[z_idx] = torch.from_numpy(
                    X[m][np.random.randint(X[m].shape[0], size=self.batch_size)]
                ).float()
            z = z.to(self.device)

            t, xt, ut, *eps = self.FM.sample_location_and_conditional_flow(
                z, rem_zts[k:k+3], return_noise=return_noise
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps[0])  # type: ignore

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 3):
                k1 = k + 1
                k2 = k + 2
                curr_ab = torch.Tensor(rem_zts[[k, k2]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab
    ############################

    ######    TRAINING    ######
    def _train(self, X, pbar=None):
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        print('Training...')

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            ## eps, ab is None if OT, float if SB
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            xt_t = torch.cat([xt, t[:, None]], dim=-1)
            vt = self.model(xt_t)
            loss = torch.mean((vt - ut) ** 2)
            flow_losses[i] = loss.item()

            if self.is_sb:
                lambda_t = self.FM.compute_lambda(t, ab)
                ## check that all lambda_t values are valid (no nans and no infs)
                assert not torch.any(torch.isnan(lambda_t) | torch.isinf(lambda_t))
                st = self.score_model(xt_t)  # type: ignore
                score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                score_losses[i] = score_loss.item()  # type: ignore
                loss += score_loss

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore
                    commit=False
                )

            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'n_steps'   : self.step_counter
                }
            )  ## commit loss to wandb servers

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################


## TODO
## Implement MM Agent which can handle any valid K
class MultiMarginalAgent(BaseAgent):
    def __init__(self, args, zt_rem_idxs, dim, K, device='cpu'):
        self.t_sampler = args.t_sampler
        self.spline = args.spline
        self.monotonic = args.monotonic
        super().__init__(args, zt_rem_idxs, dim, device=device)
        self.K = K
        self.ztdiff = np.diff(self.zt)
        self.n_windows = self.zt.shape[0] - self.K + 1

    def _build_FM(self):
        if not self.is_sb:    ## OT
            FM = MultiMarginalExactOptimalTransportConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, device=self.device
            )
        else:            ## SB
            FM = MultiMarginalSchrodingerBridgeConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, diff_ref=self.diff_ref,
                method=self.method, device=self.device
            )

        return FM

    def _get_batch(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None

        for t_idx, k in enumerate(range(self.n_windows)):
            z = torch.zeros(self.K, self.batch_size, self.dim)
            for z_idx, i in enumerate(range(k, k+self.K)):
                z[z_idx] = torch.from_numpy(
                    X[i][np.random.randint(X[i].shape[0], size=self.batch_size)]
                ).float()
            z = z.to(self.device)

            t, xt, ut, *eps = self.FM.sample_location_and_conditional_flow(
                z, self.zt[k:k+self.K], return_noise=return_noise
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps[0])  # type: ignore

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(self.n_windows):
                kend = k + self.K - 1
                i = self.batch_size * k
                j = self.batch_size * (k + 1)
                curr_ab = torch.Tensor(self.zt[[k, kend]])
                ab[i:j] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab

    #### BROKEN ####
    def _get_batch_rand_heldout(self, X, return_noise=False):
        t_heldout = np.random.randint(1, len(X) - 1)  # heldout idx
        rem_margs = np.delete(np.arange(len(X), dtype=int), t_heldout)  # list of remaining idxs
        rem_zts = self.zt[rem_margs]

        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        for k in range(len(X) - 3):
            z = torch.zeros(3, self.batch_size, self.dim)
            for z_idx, i in enumerate(range(k, k+3)):
                m = rem_margs[i]
                z[z_idx] = torch.from_numpy(
                    X[m][np.random.randint(X[m].shape[0], size=self.batch_size)]
                ).float()
            z = z.to(self.device)

            t, xt, ut, *eps = self.FM.sample_location_and_conditional_flow(
                z, rem_zts[k:k+3], return_noise=return_noise
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps[0])  # type: ignore

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 3):
                k1 = k + 1
                k2 = k + 2
                curr_ab = torch.Tensor(rem_zts[[k, k2]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab
    ############################

    ######    TRAINING    ######
    def _train(self, X, pbar=None):
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        print('Training...')
        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            ## eps, ab is None if OT, float if SB
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            xt_t = torch.cat([xt, t[:, None]], dim=-1)
            vt = self.model(xt_t)
            loss = torch.mean((vt - ut) ** 2)
            flow_losses[i] = loss.item()

            if self.is_sb:
                lambda_t = self.FM.compute_lambda(t, ab)
                ## check that all lambda_t values are valid (no nans and no infs)
                assert not torch.any(torch.isnan(lambda_t) | torch.isinf(lambda_t))
                st = self.score_model(xt_t)  # type: ignore
                score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                score_losses[i] = score_loss.item()  # type: ignore
                loss += score_loss

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore
                    commit=False
                )

            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'n_steps'   : self.step_counter
                }
            )  ## commit loss to wandb servers

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################
########################################################################
########################################################################

def build_agent(agent_type, args, zt_rem_idxs, dim, *extra_args, device='cpu'):
    ## agent_type := 'pairwise' | 'triplet' | 'mm'
    args_to_pass = ()  ## stochpy passes in live_traj_data as extra_args, but want to ignore if not tripletlive

    if agent_type == 'pairwise':
        agent_class = PairwiseAgent
    elif agent_type == 'triplet':
        agent_class = TripletAgent
    elif agent_type == 'mm':
        raise NotImplementedError (f'{agent_type} currently not implemented. Please use pairwise or triplet.')
        agent_class = MultiMarginalAgent
        args_to_pass = extra_args
    else:
        ## you should never see this
        raise NotImplementedError (f'{agent_type} not implemented')

    return agent_class(args, zt_rem_idxs, dim, *args_to_pass, device=device)
