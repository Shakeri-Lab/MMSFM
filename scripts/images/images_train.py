import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
from os import remove as rm
from os.path import exists
import wandb
from time import (
    time,
)

from mmsfm.multimarginal_cfm import (
    PairwiseExactOptimalTransportConditionalFlowMatcher,
    PairwiseSchrodingerBridgeConditionalFlowMatcher,
    MultiMarginalExactOptimalTransportConditionalFlowMatcher,
    MultiMarginalSchrodingerBridgeConditionalFlowMatcher
)

from scripts.images.images_utils import (
    load_data,
    get_hypers,
    build_models,
    get_slurm_remtime,
    RetCode,
)

CKPT_SAFETY = 3  ## safety factor for estimated time until next ckpt
### TODO: Remove if confident no NaNs will occur during training!
torch.autograd.set_detect_anomaly(True)  # type: ignore


def build_optimizer_and_scheduler(
    dataname,
    size,
    model,
    score_model,
    lrmin,
    lrmax,
    epochs,
    n_steps_epoch,
    total_iters_inc
):
    params = list(model.parameters())
    if score_model is not None:
        params += list(score_model.parameters())

    opt = torch.optim.Adam(
        params=params,
        lr=lrmax,
    )

    n = epochs * n_steps_epoch
    total_iters_dec = n - total_iters_inc

    if dataname == 'imagenette' and size == 64:
        sch = torch.optim.lr_scheduler.ConstantLR(
            opt,
            factor=1.,
            total_iters=1
        )
    else:
        sch = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [
                torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=lrmin/lrmax,
                    end_factor=1.,
                    total_iters=total_iters_inc-1,
                    last_epoch=-1
                ),
                torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=1.,
                    end_factor=lrmin/lrmax,
                    total_iters=n-total_iters_inc,
                    last_epoch=-1
                )
            ],
            [total_iters_inc-1],
            last_epoch=-1
        )

    return opt, sch


def build_FM(
        K,
        sm,
        zt,
        sigma,
        spline,
        monotonic,
        method,
        t_sampler,
        diff_ref,
        device
):
    if K == 1 and sm:
        FM = PairwiseSchrodingerBridgeConditionalFlowMatcher(
            zt=zt, sigma=sigma, diff_ref=diff_ref, ot_method=method
        )
    elif K == 1 and not sm:
        FM = PairwiseExactOptimalTransportConditionalFlowMatcher(
            zt=zt, sigma=sigma
        )
    elif K > 1 and sm:
        FM = MultiMarginalSchrodingerBridgeConditionalFlowMatcher(
            sigma=sigma, spline=spline, monotonic=monotonic,
            t_sampler=t_sampler, diff_ref=diff_ref,
            method=method, device=device
        )
    else:
        FM = MultiMarginalExactOptimalTransportConditionalFlowMatcher(
            sigma=sigma, spline=spline, monotonic=monotonic,
            t_sampler=t_sampler, device=device
        )

    return FM


def get_batch(
        FM,
        X,
        batch_size,
        dims,
        zt,
        sm,
        K=2,
        device='cpu'
):
    ts = []
    xts = []
    uts = []
    noises = []
    ztdiff = np.diff(zt)

    for t_idx, k in enumerate(range(len(X) - K)):
        z = torch.zeros(K+1, batch_size, *dims)
        for z_idx, i in enumerate(range(k, k+K+1)):
            n = X[i].shape[0]
            idxs = np.random.randint(n, size=batch_size)
            z[z_idx] = X[i][idxs]
        z = z.to(device)

        if K == 1:  # Pairwise
            _t = torch.rand(z[0].shape[0]).type_as(z[0])
            _t = zt[t_idx] + (_t * ztdiff[t_idx])
            t, xt, ut, *eps = FM.sample_location_and_conditional_flow(
                z[0].view(batch_size, -1), z[1].view(batch_size, -1),
                t=_t, t_start=t_idx, t_end=t_idx+1, return_noise=sm
            )
        else:       # MultiMarginal
            t, xt, ut, *eps = FM.sample_location_and_conditional_flow(
                z.view(K+1, batch_size, -1), zt[k:k+K+1], return_noise=sm
            )

        ts.append(t)
        xts.append(xt.view(batch_size, *dims))
        uts.append(ut.view(batch_size, *dims))
        if sm:
            noises.append(eps[0])

    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)

    if sm:
        noises = torch.cat(noises)
        ab = torch.Tensor(t.shape[0], 2)
        for k in range(len(X) - K):
            kend = k + K
            i = batch_size * k
            j = batch_size * (k + 1)
            curr_ab = torch.Tensor(zt[[k, kend]])
            ab[i:j] = curr_ab
        ab = ab.to(device)
    else:
        ab = None

    return t, xt, ut, noises, ab


def train(
    run,
    X,
    batch_size,
    accum_steps,
    dims,
    zt,
    epochs,
    n_steps_epoch,
    model,
    score_model,
    optimizer,
    scheduler,
    K,
    FM,
    sm,
    flow_losses,
    score_losses,
    lrs,
    progress,
    save_interval,
    ckpt_interval,
    checkpoint,
    outdir,
    device='cpu'
) -> RetCode:
    retcode = RetCode.DONE

    pbardesc = f'{epochs} Epochs ({n_steps_epoch} steps per epoch)'
    pbar = tqdm(total=epochs*n_steps_epoch, desc=pbardesc)
    pbar.n = progress * n_steps_epoch
    pbar.last_print_n = progress * n_steps_epoch
    pbar.refresh()

    for i in range(progress, epochs):
        t_start = time()
        train_epoch(
            run,
            X,
            batch_size,
            accum_steps,
            dims,
            zt,
            n_steps_epoch,
            model,
            score_model,
            optimizer,
            scheduler,
            K,
            FM,
            sm,
            flow_losses,
            score_losses,
            lrs,
            i,
            outdir,
            pbar=pbar,
            device=device
        )

        if (i + 1) % save_interval == 0:
            ## save mid-training state in case needed for future analysis
            print(f'Saving training-in-progress state ({i+1} epochs)...')
            torch.save(
                {
                    'model_state_dict'       : model.state_dict(),
                    'score_model_state_dict' : score_model.state_dict() if sm else None,
                    'progress'               : i + 1
                },
                f'{outdir}/on_training_{i+1:0>3d}.tar'
            )

        if (i + 1) % ckpt_interval == 0:
            ## save checkpoint for resume training
            print(f'Saving checkpoint after {i+1} epochs...')
            torch.save(  ## save state dicts separately to allow for torch.load(..., weights_only=True)
                {
                    'model_state_dict'       : model.state_dict(),
                    'score_model_state_dict' : score_model.state_dict() if sm else None,
                    'optimizer_state_dict'   : optimizer.state_dict(),
                    'scheduler_state_dict'   : scheduler.state_dict()
                },
                f'{outdir}/{checkpoint}_models.tar'
            )
            np.savez(  ## save other training progress info
                f'{outdir}/{checkpoint}_prog.npz',
                flow_losses=flow_losses,
                score_losses=score_losses,
                lrs=lrs,
                progress=i+1
            )

            ## check if enough time for next checkpoint
            t_ckpt = time()
            rts = get_slurm_remtime()  ## remaining time in seconds
            elapsed = t_ckpt - t_start  ## runtime for current checkpoint
            safety = elapsed * CKPT_SAFETY
            print(f'Estimated time required for next checkpoint is {safety:.2f}s')
            print(f'Remaining time in job is {rts}s')
            if safety >= rts:
                ## cannot finish next ckpt interval before time up
                # print(f'Estimated time required for next checkpoint ({safety:.2f}s) exceeds remaining time in job ({rts}s)')
                print('Estimated time required for next checkpoint exceeds remaining time in job.')
                print('Ending training loop. Please rerun a new job to resume training.')
                retcode = RetCode.RERUN
                break
            else:
                print('Continuing training until next checkpoint.')

    return retcode


def train_epoch(
    run,
    X,
    batch_size,
    accum_steps,
    dims,
    zt,
    n_steps,
    model,
    score_model,
    optimizer,
    scheduler,
    K,
    FM,
    sm,
    flow_losses,
    score_losses,
    lrs,
    epoch_num,
    outdir,
    pbar=None,
    device='cpu'
):
    ## fills in flow losses and score losses arrays in place
    maybe_close = False
    if pbar is None:
        maybe_close = True
        pbar = tqdm(total=n_steps)

    curr_grad_step = epoch_num * n_steps
    for i in range(n_steps):
        optimizer.zero_grad()
        ## gradient accumulation for effective batch size of batch_size * accum_steps
        for accum_i in range(accum_steps):
            t, xt, ut, eps, ab = get_batch(FM, X, batch_size, dims, zt, sm, K=K, device=device)
            try:  ## check for nans in inputs, outputs, and loss
                assert not torch.any(torch.isnan(t) | torch.isinf(t))
                assert not torch.any(torch.isnan(xt) | torch.isinf(xt))
                assert not torch.any(torch.isnan(ut) | torch.isinf(ut))
                assert not torch.any(torch.isnan(eps) | torch.isinf(eps))  # type: ignore
                assert not torch.any(torch.isnan(ab) | torch.isinf(ab))  # type: ignore

                vt = model(t, xt)
                assert not torch.any(torch.isnan(vt) | torch.isinf(vt))
                loss = torch.mean((vt - ut) ** 2)
                flow_losses[epoch_num, i] += loss.item()

                if sm:
                    lambda_t = FM.compute_lambda(t, ab)
                    ## check that all lambda_t values are valid (no nans and no infs)
                    assert not torch.any(torch.isnan(lambda_t) | torch.isinf(lambda_t))

                    st = score_model(t, xt).view(eps.shape)  # type: ignore
                    assert not torch.any(torch.isnan(st) | torch.isinf(st))

                    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                    score_losses[epoch_num, i] += score_loss.item()
                    loss += score_loss

                loss /= accum_steps
                loss.backward()
            except:
                last_valid = {
                    'model_state_dict' : model.state_dict(),
                    'score_model_state_dict' : score_model.state_dict() if sm else None,
                    't' : t,
                    'xt' : xt,
                    'ut' : ut,
                    'eps' : eps,
                    'ab' : ab,
                    'vt' : vt,  # type: ignore
                    'st' : st if sm else None,  # type: ignore
                    'loss' : loss,  # type: ignore
                    'curr_grad_step' : curr_grad_step,
                    'accum_i' : accum_i,
                }
                last_valid_outdir = f'{outdir}/last_valid.pt'
                print(f'Saving last valid training step info to {last_valid_outdir}...')
                torch.save(last_valid, last_valid_outdir)
                raise

        optimizer.step()
        prev_lr = scheduler.get_last_lr()[0]
        lrs[epoch_num, i] = prev_lr
        scheduler.step()

        flow_losses[epoch_num, i] /= accum_steps

        if sm: ## maybe log score loss
            score_losses[epoch_num, i] /= accum_steps
            run.log({'score_loss' : score_losses[epoch_num, i]}, commit=False)

        ## finally commit log to wandb
        run.log({
            'flow_loss'  : flow_losses[epoch_num, i],
            'lrs'        : prev_lr,
            'grad_step'  : curr_grad_step
        })

        curr_grad_step += 1
        pbar.update(1)

    if maybe_close:
        pbar.close()


def main(args, run) -> RetCode :
    dataname = args.dataname
    size = args.size
    K = args.window_size
    spline = args.spline
    monotonic = args.monotonic
    sm = args.sm
    method = args.method
    zt = args.zt
    progression = args.progression
    batch_size = args.batch_size
    accum_steps = args.accum_steps
    e_batch_size_per_window = args.e_batch_size_per_window
    e_batch_size = args.e_batch_size
    epochs = args.n_epochs
    n_steps_epoch = args.n_steps
    sigma = args.sigma
    t_sampler = args.t_sampler
    diff_ref = args.diff_ref
    lrmin = args.lr[0]
    lrmax = args.lr[1]
    total_iters_inc = args.total_iters_inc
    resume = args.resume
    save_interval = args.save_interval
    ckpt_interval = args.ckpt_interval
    checkpoint = args.checkpoint
    device = args.device
    outdir = args.outdir

    print('Effective Batch Size:', e_batch_size)
    print('Effective Batch Size per Window:', e_batch_size_per_window)

    trainset, testset, classes, dims = load_data(dataname, size)  # type: ignore
    X = [trainset[i].to(device) for i in progression]
    run.config.update(
        {
            'progression_classes'  : [classes[i] for i in progression],
            'effective_batch_size' : e_batch_size,
            'ckpt_safety_factor'   : CKPT_SAFETY,
        },
        allow_val_change=True,
    )
    print(' -> '.join([classes[i] for i in progression]))

    FM = build_FM(K, sm, zt, sigma, spline, monotonic, method, t_sampler, diff_ref, device)
    hypers = get_hypers(dataname, size, dims)
    model, score_model = build_models(hypers, sm, device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        dataname,
        size,
        model,
        score_model,
        lrmin,
        lrmax,
        epochs,
        n_steps_epoch,
        total_iters_inc
    )

    if resume:
        print('Loading checkpoint...')
        ckpt_models = torch.load(f'{outdir}/{checkpoint}_models.tar')
        model.load_state_dict(ckpt_models['model_state_dict'])
        if sm:
            score_model.load_state_dict(ckpt_models['score_model_state_dict'])  # type: ignore
        optimizer.load_state_dict(ckpt_models['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt_models['scheduler_state_dict'])

        with np.load(f'{outdir}/{checkpoint}_prog.npz') as ckpt_progress:
            flow_losses = ckpt_progress['flow_losses']
            score_losses = ckpt_progress['score_losses'] if sm else None
            lrs = ckpt_progress['lrs']
            progress = int(ckpt_progress['progress'])  ## number of epochs trained so far

    else:
        print('Initializing new training...')
        flow_losses = np.zeros((epochs, n_steps_epoch))
        score_losses = np.zeros_like(flow_losses) if sm else None
        lrs = np.zeros_like(flow_losses)
        progress = 0

    ## Set models in training mode
    ## and
    ## configure wandb for logging per gradient step
    ## and
    ## configure wandb to watch model gradients and parameters
    run.define_metric('grad_step')
    run.define_metric('flow_loss', step_metric='grad_step')
    run.define_metric('lrs', step_metric='grad_step')
    model.train()
    nets = [model]
    if sm:
        run.define_metric('score_loss', step_metric='grad_step')
        score_model.train()  # type: ignore
        nets.append(score_model)

    run.watch(nets, log='all')

    retcode = train(
        run,
        X,
        batch_size,
        accum_steps,
        dims,
        zt,
        epochs,
        n_steps_epoch,
        model,
        score_model,
        optimizer,
        scheduler,
        K,
        FM,
        sm,
        flow_losses,
        score_losses,
        lrs,
        progress,
        save_interval,
        ckpt_interval,
        checkpoint,
        outdir,
        device=device
    )

    if retcode is RetCode.DONE:
        ## cleanup tmp ckpt files and save final models+losses upon completing training
        print('Saving models and losses...')
        torch.save(model.state_dict(), f'{outdir}/flow_model.pth')
        np.save(f'{outdir}/flow_losses.npy', flow_losses)

        if sm:
            torch.save(score_model.state_dict(), f'{outdir}/score_model.pth')  # type: ignore
            np.save(f'{outdir}/score_losses.npy', score_losses)  # type: ignore

        np.save(f'{outdir}/lrs.npy', lrs)

        ## TODO: uncomment if ckpt files are unwanted after training
        # print('Cleaning up checkpoint files...')
        # ckpt_files = [
            # f'{outdir}/{checkpoint}_models.tar',
            # f'{outdir}/{checkpoint}_prog.npz'
        # ]
        # for fpath in ckpt_files:
            # if exists(fpath):
                # rm(fpath)

    return retcode

