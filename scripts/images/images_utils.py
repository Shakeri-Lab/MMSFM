import subprocess
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
import joblib
import os
from enum import Enum

from torchcfm.models.unet import UNetModel  # type: ignore


class RetCode(Enum):
    DONE = 0  ## Done training
    RERUN = 1  ## Need to rerun code from timeout-based premature exit, only used in imagenette_train
    ## No code for failures/exceptions to not mess with stack trace


def slurmtime2sec(slurmtime: str) -> int:
    """Converts slurm time format to seconds
    Convert string of [dd-][hh:][mm:]ss to list of dd, hh, mm, ss.
    Then convert values to seconds and return sum.

    :slurmtime: str : [dd-][hh:][mm:]ss
    :returns: int

    """
    dhms = slurmtime.split('-')  ## split dd from hh:mm:ss
    if len(dhms) == 1:  ## edge case where time looks like [hh:][mm:]ss
        d = '0'
        hms = dhms[0]
    else:
        d, hms = dhms[0], dhms[1]
    hms = hms.split(':')  ## [hh, mm, ss]
    td = int(d) * 86400  ## 60 * 60 * 24 = 84600
    thms = sum([(60**i)*int(t) for i, t \
                in enumerate(hms[::-1])])  ## enumerate from ss -> mm -> hh
    return td + thms


def get_slurm_remtime() -> int:
    """Get time remaining for current slurm job
    Retrieve output of "squeue -h -j $SLURM_JOB_ID -o %L"
    which comes in the form [dd-][hh:][mm:]ss
    and converts it to seconds.

    If $SLURM_JOB_ID does not exist, assume the process
    is not running from a slurm job and return the number
    of seconds in 999 days.

    :returns: int

    """
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is None:
        ## not running from a slurm job. Always return time in sec for 999 days
        ## 999 * 24 * 60 * 60 = 86313600
        print('No slurm job detected. Defaulting remaining time to 999 days')
        return 86313600
    else:
        cmd = ['squeue', '-h', '-j', slurm_job_id, '-o', '%L']
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        remtime = result.stdout.decode('utf-8').strip()
        return slurmtime2sec(remtime)


def load_cifar10_by_class():
    trainset = joblib.load(f'./data/cifar10/trainset.pkl')
    testset = joblib.load(f'./data/cifar10/testset.pkl')
    dims = trainset[0].shape[-3:]

    classes = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    return trainset, testset, classes, dims


def load_imagenette_by_class(size):
    trainset = joblib.load(f'./data/imagenette/trainset{size}.pkl')
    testset = joblib.load(f'./data/imagenette/testset{size}.pkl')
    dims = trainset[0].shape[-3:]

    classes = (
        'tench', 'English springer', 'cassette player',
        'chain saw', 'church', 'French horn', 'garbage truck',
        'gas pump', 'golf ball', 'parachute'
    )

    return trainset, testset, classes, dims


def load_data(dataname, size):
    if dataname == 'imagenette':
        return load_imagenette_by_class(size)

    elif dataname == 'cifar10':
        return load_cifar10_by_class()


def get_hypers(dataname, size, dims):
    hypers = {}
    if dataname == 'imagenette':
        hypers['dims'] = dims
        if size == 32:
            hypers['channels'] = 256
            hypers['depth'] = 3
            hypers['channel_mult'] = (1, 2, 2, 2)
            hypers['attention_res'] = '16,8'
            hypers['use_fp16'] = False
        elif size == 64:
            hypers['channels'] = 192
            hypers['depth'] = 3
            hypers['channel_mult'] = (1, 2, 3, 4)
            hypers['attention_res'] = '32,16,8'
            hypers['use_fp16'] = True
        elif size == 128:
            hypers['channels'] = 256
            hypers['depth'] = 3
            hypers['channel_mult'] = (1, 1, 2, 3, 4)
            hypers['attention_res'] = '32,16,8'
            hypers['use_fp16'] = True
    elif dataname == 'cifar10':
        hypers['dims'] = dims
        hypers['channels'] = 256
        hypers['depth'] = 2
        hypers['channel_mult'] = (1, 2, 2, 2)
        hypers['attention_res'] = '16'
        hypers['use_fp16'] = False

    return hypers


def build_models(hypers, sm, device):
    dims = hypers['dims']
    channels = hypers['channels']
    depth = hypers['depth']
    channel_mult = hypers['channel_mult']  ## tuple of ints
    attention_res = hypers['attention_res']  ## string of ints sep. by commas
    use_fp16 = hypers['use_fp16']

    model = UNetModel(
        dim=dims,
        num_channels=channels,
        num_res_blocks=depth,   ## depth?
        channel_mult=channel_mult,
        learn_sigma=False,
        class_cond=False,
        num_classes=None,
        use_checkpoint=False,
        attention_resolutions=attention_res,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0.,
        resblock_updown=False,
        use_fp16=use_fp16,
        use_new_attention_order=False,
    ).to(device)

    if sm:
        score_model = UNetModel(
            dim=dims,
            num_channels=channels,
            num_res_blocks=depth,
            channel_mult=channel_mult,
            learn_sigma=False,
            class_cond=False,
            num_classes=None,
            use_checkpoint=False,
            attention_resolutions=attention_res,
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            dropout=0,
            resblock_updown=False,
            use_fp16=use_fp16,
            use_new_attention_order=False,
        ).to(device)
    else:
        score_model = None

    return model, score_model

