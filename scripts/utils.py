import datetime
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from torch.cuda import is_available as cuda_is_available
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

from MIOFlow.utils import group_extract  # type: ignore
from MIOFlow.datasets import (           # type: ignore
    make_diamonds,    ## petals
    make_dyngen_data  ## dyngen
)

import wandb


class IndexableScaler(ABC):
    '''Indexable Scaler

    After fitting scaler, can transform input X w/ user-defined indexes.
    Allows inputs X with shape (M, N, D)
    where
    M := number of timepoint marginals
    N := number of samples
    D := number of features

    Is flexible enough to allow X := np.ndarray of shape (M, N, D)
    or
    X := list of M np.ndarrays of shape (N, D)
    '''
    def __init__(self, scaler, name):
        self.fitted = False
        self.scaler = scaler
        self.name = name

    def fit(self, X, y=None):
        '''Fits scaler to input data X.

        X is list-like of np.ndarrays with shape (N_i, D)
        where N_i is number of samples in the i_th array
        OR an np.ndarray of shape (M, N, D)
        where M is the number of marginals
        and N is the number of samples per marginal

        y is ignored
        '''
        self.scaler.fit(np.vstack(X))
        self.fitted = True

    def transform(self, X, y=None, idxs=None):
        '''Transforms X using the fitted mean, scale (var) parameters.

        X is list-like of np.ndarrays with shape (N_i, D)
        where N_i is number of samples in the i_th array
        OR an np.ndarray of shape (M, N, D)
        where M is the number of marginals
        and N is the number of samples per marginal

        If idxs is None, then transform all features of X.

        If idxs is a list-like of feature indices into X,
        then assume X has len(idxs) features and only transform using
        the parameters selected by idxs.
        '''
        if not self.fitted:
            raise RuntimeError(f'{self.name} has not been fitted!')

        ## y is ignored
        if idxs is None:
            Xscaled = [self.scaler.transform(X_i) for X_i in X]
        else:
            Xscaled = [self._index_transform(X_i, idxs) for X_i in X]

        if isinstance(X, np.ndarray):
            Xscaled = np.asarray(Xscaled)

        return Xscaled

    @abstractmethod
    def _index_transform(self, Xsubset, idxs):
        '''Transforms Xsubset by the scaler parameters selected by idxs

        idxs is a list-like of integers pointing to columns of X in Xsubset.
        '''
        raise NotImplementedError('Not Implemented for Abstract Class')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y)
        return self.transform(X, y=y, **fit_params)

    def inverse_transform(self, X):
        '''Inverse-Transforms X back to original domain.'''
        if not self.fitted:
            raise RuntimeError(f'{self.name} has not been fitted!')

        Xunscaled = [self.scaler.inverse_transform(X_i) for X_i in X]

        if isinstance(X, np.ndarray):
            Xunscaled = np.stack(Xunscaled)

        return Xunscaled


class IndexableNoOpScaler(IndexableScaler):
    '''No Op Scaler
    Does not scale and only passes through outputs.
    '''

    class _NoOpScaler():
        '''No Op Scaler
        Only to hack in no-op polymorphism with sklearn scalers
        '''

        def fit(self, X):
            pass

        def transform(self, X):
            return X[:, :]

        def inverse_transform(self, X):
            return X[:, :]

    def __init__(self):
        super().__init__(
            self._NoOpScaler(),
            'IndexableNoOpScaler',
        )

    def _index_transform(self, Xsubset, idxs):
        '''No Op Transform

        Simply passes through Xsubset
        '''
        return Xsubset[:, :]


class IndexableStandardScaler(IndexableScaler):
    '''Indexable Standard Scaler
    Scales data using a standard scale s.t. scaler(X) : X -> R
    where mean(R) = 0 and var(R) = 1.
    '''
    def __init__(self):
        super().__init__(
            StandardScaler(),  ## automatically handles cases where var == 0
            'IndexableStandardScaler',
        )

    def _index_transform(self, Xsubset, idxs):
        '''Computes Standard Scaling z = (x - mu) / std'''
        tmp = (Xsubset - self.scaler.mean_[None, idxs])
        return tmp / self.scaler.scale_[None, idxs]


class IndexableMinMaxScaler(IndexableScaler):
    '''Indexable Min-Max Scaler
    Scales data using a minmax scale s.t. scaler(X) : X -> [0, 1]
    '''
    def __init__(self):
        super().__init__(
            MinMaxScaler(),  ## automatically handles cases where var == 0
            'IndexableMinMaxScaler',
        )

    def _index_transform(self, Xsubset, idxs):
        '''Computes MinMax Scaling z = (x - xmin) / (xmax - xmin)

        Uses scaler.scale_ := 1 / (xmax - xmin) and
             scaler.min_   := -xmin * scaler.scale_
             ==> z = x * scaler.scale_ + scaler.min_
        '''
        tmp = Xsubset * self.scaler.scale_[None, idxs]
        return tmp + self.scaler.min_[None, idxs]


########################     TIMER DECORATOR    ########################
########################################################################
def timer_func(f):
    def wrap_func(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        print(f'Function {f.__name__!r} executed in {((t1 - t0) / 60):.4f}m ({(t1 - t0):.4f}s)')
        return result
    return wrap_func
########################################################################
########################################################################
def load_data(dataname):
    print('Loading data...')
    datalabels = None
    testdatalabels = None
    margtimes = None
    if dataname == 'petals':
        petals = make_diamonds()
        data = []
        for group in np.sort(petals.samples.unique()):
            petals_i = group_extract(petals, group)
            data.append(petals_i)
        testdata = data

    elif dataname == 'dyngen':
        dyngen = make_dyngen_data(phate_dims=5)
        data = []
        for group in np.sort(dyngen.samples.unique()):
            dyngen_i = group_extract(dyngen, group)
            data.append(dyngen_i)
        testdata = data

    elif dataname[:4] == 'cite' or dataname[:4] == 'mult':
        basename, variant = dataname.split('_')
        loadname = f'{basename}_{variant}'

        train_npz = np.load(f'data/{loadname}_train.npz')
        test_npz = np.load(f'data/{loadname}_test.npz')

        data = [train_npz[f't{i}'] for i in range(4)]
        testdata = [test_npz[f't{i}'] for i in range(4)]

    elif dataname == 'sg':
        data = np.load('data/sg.npy')
        testdata = np.load('data/sgtest.npy')

    elif dataname == 'alphag':
        data = np.load('data/alphag.npy')
        testdata = np.load('data/alphagtest.npy')

    elif dataname[1] == 'g':
        data = np.load('data/3g.npy')
        testdata = data

    else:
        data = np.load('data/gcm.npy')
        testdata = data

    print('Loaded data!')

    return data, testdata, datalabels, testdatalabels, margtimes


def select_marginals(dataname):
    if dataname == 'petals' or dataname == 'dyngen':
        marginals = [0, 1, 2, 3, 4]
    elif dataname[:4] == 'cite' or dataname[:4] == 'mult':
        marginals = [0, 1, 2, 3]
    elif dataname == 'sg' or dataname == 'alphag':
        marginals = [0, 1, 2, 3, 4, 5, 6]
    elif dataname == '2g':
        marginals = [0, 1]
    elif dataname == '3g':
        marginals = [0, 1, 2]
    elif dataname == 'gc':
        marginals = [0, 1]
    elif dataname == 'gm':
        marginals = [0, 2]
    elif dataname == 'cm':
        marginals = [1, 2]
    elif dataname == 'gcm':
        marginals = [0, 1, 2]
    else:
        raise ValueError('Must select valid dataname. You should never see this.')

    return marginals


def load_data_and_marginals(dataname):
    data, testdata, datalabels, testdatalabels, margtimes = load_data(dataname)
    marginals = select_marginals(dataname)

    return data, testdata, datalabels, testdatalabels, margtimes, marginals


def build_zt(zt, marginals):
    ## timepoint labels zt := list | None
    ## number of marginals in data m := int | None
    ## eval timepoint labels eval_zt := list | None

    ## build zt
    if zt is None and marginals is not None:
        m = len(marginals)
        # equidistant zt
        zt = np.linspace(0, 1, m)

    elif isinstance(zt, list):
        # arbitrary zt
        zt = np.array(zt)

        if not np.all(np.diff(zt) > 0):
            # enforce strictly increasing
            raise ValueError('Specified marginal timepoints (--zt) are not strictly increasing.')

        if isinstance(marginals, list) and zt.shape[0] != len(marginals):
            # check if number of timepoints match number of marginals
            raise ValueError('Number of specified marginal timepoints (--zt) do not match the number of marginals.')

    else:
        ## should never happen
        raise ValueError('Need to specify either marginal timepoints (--zt), number of marginals, or both.')

    ## normalize zt to the range [0, 1] and scale eval_zt by same weight and bias
    a = zt[0]
    b = zt[-1]
    zt = (zt - a) / (b - a)
    return zt


def set_up_exp(args):
    ## Sets up output directory and records arguments for the experiment
    if args.outdir == None:
        outdir = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-4]
    else:
        outdir = args.outdir
    outdir = f'results/{outdir}'

    os.makedirs(outdir, exist_ok=True)

    ## write args used to file for documentation
    with open(f'{outdir}/args.txt', 'w') as f:
        for k, v in vars(args).items():
            if k == 'zt':
                v = np.round(v, decimals=4).tolist()
            f.write(f'{k: <27} = {v}\n')

    return outdir


def update_eval_losses_dict(acc, curr):
    for k, v in curr.items():
        acc_v = acc.get(k, None)
        if acc_v is None:
            acc_v = v.reshape((1, -1))
        else:
            acc_v = np.vstack((acc_v, v))
        acc[k] = acc_v
    return acc


def write_eval_losses(eval_losses_dict, outdir, is_sb=False):
    suffix = 'sde' if is_sb else 'ode'
    outdirname = f'{outdir}/eval_losses_{suffix}.npz'
    np.savez(outdirname, **eval_losses_dict)


def report_heldout_evals(eval_losses_dict, eval_zt_idx, outdir, is_sb=False):
    lines = []
    diffeq_type = 'SDE' if is_sb else 'ODE'
    title = f'{diffeq_type} Eval Losses at idx(s) {eval_zt_idx}'
    titleline = '-'*8 + f' {title} ' + '-'*8
    lines.append(titleline)
    for loss_name, losses in eval_losses_dict.items():
        line = f'{loss_name: <16} : {losses[-1, eval_zt_idx]}'
        lines.append(line)
    closeline = '-'*len(titleline)
    lines.append(closeline)

    ## print lines
    for line in lines:
        print(line)

    ## and also save to a txt for quick reading
    outdirname = f'{outdir}/heldout_evals.txt'
    with open(outdirname, 'a') as f:
        ## writelines() does not automatically add newlines...
        f.writelines(map(lambda x: x + '\n', lines))


def unconcatenate_like(concatdata, target):
    margidxs = [target[0].shape[0]]
    for i in range(1, len(target)):
        margidxs.append(margidxs[i-1] + target[i].shape[0])

    ## final element is empty array so don't include that
    return np.split(concatdata, margidxs)[:-1]


def get_device(nogpu):
    if not nogpu:
        cuda_avail = cuda_is_available()
        if not cuda_avail:
            print('Requested gpu but gpu not available. Defaulting to cpu...')
            device = 'cpu'
        else:
            print('Using cuda...')
            device = 'cuda'
    else:
        print('Using cpu...')
        device = 'cpu'

    return device


def format2df(data, zt):
    ## Used for converting my data format into MIOFlow data format
    ## data should be list[np.ndarray] where each np.ndarray has shape (N, dim)
    ## OR np.ndarray with shape (M, N, dim) where M is the number of marginals

    dfs = [pd.DataFrame(data[i]) for i in range(len(data))]
    colnames = [f'd{i+1}' for i in range(data[0].shape[1])]
    for df, t in zip(dfs, zt):
        df.columns = colnames
        df['samples'] = t

    dfall = pd.concat(dfs, ignore_index=True)

    return dfall


def get_run_id(entity, project, run_name):
    runpath = f'{entity}/{project}'
    api = wandb.Api()
    runs = api.runs(
        path=runpath,
        filters={'display_name' : {'$regex' : run_name}}
    )

    if len(runs) == 0:
        print(f'No run found with name "{run_name}". Initializing new run.')
        run_id = None
    else:
        if len(runs) == 1:
            print(f'Resuming run "{run_name}".')
        else:
            ## maybe should never happen?
            print(f'Multiple runs found with name "{run_name}". Resuming most recent run.')
        run_id = runs[0].id

    return run_id

