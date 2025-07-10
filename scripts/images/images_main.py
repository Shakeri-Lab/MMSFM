import argparse
import os
from time import (
    sleep,
)
import datetime
import numpy as np
from torch.cuda import (
    is_available as cuda_is_available,
    device_count as cuda_device_count,
    get_device_properties as cuda_get_device_properties
)
import wandb

from scripts.images import (
    images_train,
    images_eval,
    images_plot,
)
from scripts.images.images_utils import (
    RetCode,
)

RESUBMITFILE = os.getenv('RESUBMITFILE', 'RESUBMIT')
EXCEPTIONFILE = os.getenv('EXCEPTIONFILE', 'EXCEPTION')


def int_or_float(x):
    try:
        return int(x)
    except:
        return float(x)


def set_up_exp(args):
    ## Sets up output directory and records arguments for the experiment
    os.makedirs(args.outdir, exist_ok=True)

    ## write args used to file for documentation
    with open(f'{args.outdir}/args.txt', 'w') as f:
        for k, v in vars(args).items():
            if k == 'zt':
                v = np.round(v, decimals=4).tolist()
            f.write(f'{k: <27} = {v}\n')

    ## if resubmit flag from previous call exists, remove it
    resubmitflag = os.path.join(args.outdir, RESUBMITFILE)
    if os.path.exists(resubmitflag):
        print('Removing resubmit flag...')
        os.remove(resubmitflag)

    ## if exception flag from previous call exists, remove it
    exceptionflag = os.path.join(args.outdir, EXCEPTIONFILE)
    if os.path.exists(exceptionflag):
        print('Removing exception flag...')
        os.remove(exceptionflag)

    return resubmitflag, exceptionflag


def chk_fmt_args(args):
    #### Checks and Formats args in-place
    VALID_CMDS = ['train', 'eval', 'plot']
    args.cmds = [cmd.lower() for cmd in args.cmds]
    for cmd in args.cmds:
        assert cmd in VALID_CMDS, f'"{cmd} is not a valid command. Choose "train", "eval", or "plot"'

    ## check that window_size is valid
    assert args.window_size >= 1, 'Invalid window size detected. Please ensure K >= 1'

    ## check that progression is supplied
    assert args.progression is not None, 'Progression of classes is missing.'

    ## check and format zt by normalizing to [0, 1]
    if args.zt == None:
        args.zt = np.linspace(0, 1, len(args.progression))
    else:
        assert len(args.zt) == len(args.progression), 'Length of progression and number of timepoints do not match.'
        zt = np.array(args.zt)
        zt -= zt[0]
        zt /= zt[-1]
        args.zt = zt

    assert args.batch_size > 0, 'Batch size must be a positive integer'
    assert args.accum_steps > 0, 'Gradient accumulation steps must be a positive integer'

    assert args.e_batch_size_per_window is None, 'Do not manually set e_batch_size_per_window, it is programatically calculated'
    args.e_batch_size_per_window = args.batch_size * args.accum_steps

    assert args.e_batch_size is None, 'Do not manually set e_batch size, it is programatically calculated'
    n_windows = args.zt.shape[0] - args.window_size
    args.e_batch_size = args.e_batch_size_per_window * n_windows

    ## check lr provides lrmin and lrmax
    assert len(args.lr) <= 2, 'Please provide either a single lr or a min max lr pair'
    if len(args.lr) == 1:
        args.lr = args.lr * 2  ## repeat provided lr for lrmin == lrmax
    else:
        assert args.lr[0] <= args.lr[1], 'min lr > max lr'

    ## check total_iters_inc is float [0, 1] or int [2, n_epochs * n_steps - 1]
    n = args.n_epochs * args.n_steps
    total_iters_inc = args.total_iters_inc
    if isinstance(total_iters_inc, float):
        assert 0. <= total_iters_inc and total_iters_inc <= 1., 'If a float, 0 <= total_iters_inc <= 1'
        total_iters_inc = int(n * total_iters_inc)
        ## clamp to [2, n-1]
        args.total_iters_inc = max(min(total_iters_inc, n-1), 2)
    else:
        assert 2 <= total_iters_inc and total_iters_inc <= n-1, 'If an int, 2 <= total_iters_inc <= total_steps - 1'

    assert args.save_interval > 0, 'Model mid-training progress saving interval must be a positive int'
    assert args.ckpt_interval > 0, 'Checkpointing interval must be a positive int'

    ## format outdir
    if args.outdir == None:
        outdir = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')[:-4]
    else:
        outdir = args.outdir
    args.outdir = f'results/{outdir}'

    ## check cuda is available
    if not cuda_is_available() and args.device == 'cuda':
        print('Cuda not available. Setting device to CPU.')
        args.device = 'cpu'
    print(f'Using device {args.device}')

    if cuda_is_available():
        print(f'{cuda_device_count()} cuda devices available')
        for i in range(cuda_device_count()):
            print(cuda_get_device_properties(i))


def get_run_id(args):
    runpath = f'{args.entity}/{args.project}'
    api = wandb.Api()
    runs = api.runs(
        path=runpath,
        filters={'display_name' : {'$eq' : args.run_name}}
    )

    assert len(runs) < 2, f'Ambiguous run name -- Multiple runs found with name "{args.run_name}".'
    if len(runs) == 0:
        print(f'No run found with name "{args.run_name}". Initializing new run.')
        run_id = None
    else:
        print(f'Resuming run "{args.run_name}".')
        run_id = runs[0].id

    return run_id


def wait_until_run_finished(entity, project, run_id, sleep_time=10):
    api = wandb.Api()
    run_resource = f'{entity}/{project}/{run_id}'
    while True:
        run = api.run(run_resource)
        state = run.state
        if state in ['finished', 'crashed', 'failed']:
            break  ## run finished on wandb servers
        else:
            sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmds', type=str, nargs='+',
                        choices=['train', 'eval', 'plot'])

    ## Train Args
    parser.add_argument('--dataname', '-d', type=str, required=True,
                        choices=['imagenette', 'cifar10'])
    parser.add_argument('--size', type=int, default=32,
                        choices=[32, 64, 128])
    parser.add_argument('--window_size', '-K', type=int, default=2)
    parser.add_argument('--spline', type=str, default='cubic',
                        choices=['linear', 'cubic'])
    parser.add_argument('--monotonic', action='store_true')
    parser.add_argument('--score_matching', action='store_const',
                        const=True, default=False, dest='sm')
    parser.add_argument('--method', type=str, default='exact',
                        choices=['exact', 'sinkhorn', 'unbalanced', 'partial'])
    parser.add_argument('--zt', type=float, nargs='+')
    parser.add_argument('--progression', '-p', type=int, nargs='+')
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--accum_steps', '-a', type=int, default=1)
    parser.add_argument('--e_batch_size_per_window', type=int, default=None)  ## DO NOT TOUCH! chk_fmt_args() will raise error if set
    parser.add_argument('--e_batch_size', type=int, default=None)  ## DO NOT TOUCH! chk_fmt_args() will raise error if set
    parser.add_argument('--n_steps', '-n', type=int, default=1000)
    parser.add_argument('--n_epochs', '-e', type=int, default=1)
    parser.add_argument('--sigma', '-s', type=float, default=0.15)
    parser.add_argument('--t_sampler', type=str, default='stratified',
                        choices=['uniform', 'stratified'])
    parser.add_argument('--diff_ref', type=str, default='miniflow',
                        choices=['whole', 'miniflow'])
    parser.add_argument('--lr', '-r', type=float, nargs='+', default=[1e-8, 1e-4])
    parser.add_argument('--total_iters_inc', type=int_or_float, default=0.5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='ckpt')

    ## Inference Args
    parser.add_argument('--n_infer', type=int, default=10)
    parser.add_argument('--t_infer', type=int, default=9)
    ## Only used in eval and plot scripts
    parser.add_argument('--load_models', type=str, default=None)

    ## Plot Args
    parser.add_argument('--scale', type=int, default=4)

    ## WandB Args
    parser.add_argument('--entity', type=str, default='')
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--resume', action='store_const',
                        const=True, default=False)
    parser.add_argument('--no_wandb', action='store_const',
                        const='disabled', default='online', dest='wandb_mode')
    parser.add_argument('--nogpu', action='store_const',
                        const='cpu', default='cuda', dest='device')
    parser.add_argument('--outdir', '-o', type=str, default=None)
    args = parser.parse_args()

    chk_fmt_args(args)

    if args.wandb_mode == 'online' and args.resume:
        run_id = get_run_id(args)
    else:
        run_id = None

    if run_id is None:
        args.resume = False

    resubmitflag, exceptionflag = set_up_exp(args)

    ## Get cmds (and thus scripts) sorted into train, eval, plot order
    sorted_cmds = []
    for validcmd in ['train', 'eval', 'plot']:
        if validcmd in args.cmds:
            sorted_cmds.append(validcmd)

    scripts = []
    for cmd in sorted_cmds:
        if cmd == 'train':
            scripts.append(images_train.main)
        if cmd == 'eval':
            scripts.append(images_eval.main)
        if cmd == 'plot':
            scripts.append(images_plot.main)

    ## Set up WandB Run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=args,  # type: ignore
        mode=args.wandb_mode,
        id=run_id,
        name=args.run_name,
        resume='allow'
    )
    ## update run_id variable if necessary
    run_id = run.id

    try:
        ## run scripts in sorted_cmds order
        for cmdname, script in zip(sorted_cmds, scripts):
            ## only train can return RetCode.RERUN
            ## eval and plot always return RetCode.DONE
            print(f'Running {cmdname} script...')
            retcode = script(args, run)
            if retcode is RetCode.RERUN:
                ## break here to exit with block and conclude run
                ## also prevents calling eval and plot scripts
                ## on not fully trained model
                break

        run.finish()

        if retcode is RetCode.RERUN:  # type: ignore
            ## set resubmit flag for resuming training later
            print('Writing resubmit flag')
            with open(resubmitflag, 'w') as f:
                f.write('1')

            print('Waiting until run finished on WandB servers...')
            ## wait a bit so wandb has time to conclude run
            wait_until_run_finished(args.entity, args.project, run_id)
            print('Confirmed run finished. Safely exiting for rerun.')
        elif retcode is RetCode.DONE:  # type: ignore
            ## all scripts finished so remove resubmit flag if it exists
            print('Run completed.')
            if os.path.exists(resubmitflag):
                os.remove(resubmitflag)
    except:
        ## something broke during training so set exception flag
        print('Exception occured. Writing exception flag')
        with open(exceptionflag, 'w') as f:
            f.write('1')
        print('Marking run as crashed on WandB...')
        run.finish(exit_code=1)
        print('Marked run as crashed. Raising exception.')
        raise


if __name__ == '__main__':
    main()
