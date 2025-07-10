import argparse

import numpy as np

from tqdm import tqdm

from modelagent import build_agent
from plotter import Plotter
from utils import (
    build_zt,
    IndexableMinMaxScaler,
    IndexableStandardScaler,
    IndexableNoOpScaler,
    get_device,
    load_data_and_marginals,
    report_heldout_evals,
    set_up_exp,
    timer_func,
    update_eval_losses_dict,
    write_eval_losses,
    get_run_id
)

import wandb

@timer_func
def main():
    parser = argparse.ArgumentParser()

    ### Training Args ###
    parser.add_argument('--dataname', '-d', type=str, required=True,
                        choices=['sg', 'alphag',
                                 '2g', 'gc', 'gm', 'cm', '3g', 'gcm',
                                 'petals', 'dyngen',
                                 'cite_pca50', 'cite_pca100', 'cite_hivars',
                                 'mult_pca50', 'mult_pca100', 'mult_hivars'])
    parser.add_argument('--scaler_type', type=str, default='minmax',
                        choices=['minmax', 'standard', 'noop'])
    parser.add_argument('--flowmatcher', '-f', type=str, required=True,
                        choices=['ot', 'sb'])
    ## TODO: add in 'mm' when modelagent.MultiMarginalAgent is implemented
    parser.add_argument('--agent_type', '-a', type=str, required=True,
                        choices=['pairwise', 'triplet'])
    parser.add_argument('--t_sampler', type=str, default='stratified',
                        choices=['uniform', 'stratified'])
    parser.add_argument('--diff_ref', type=str, default='miniflow',
                        choices=['whole', 'miniflow'])
    ## Window size is currently ignored
    parser.add_argument('--window_size', '-K', type=int, default=2)
    parser.add_argument('--spline', type=str, required=True,
                        choices=['linear', 'cubic'])
    parser.add_argument('--monotonic', action='store_true')
    parser.add_argument('--modelname', type=str, required=True,
                        choices=['mlp'])
    parser.add_argument('--modeldepth', type=int, default=2)
    parser.add_argument('--method', '-m', type=str, default='exact')
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--n_steps', '-n', type=int, default=1000)
    parser.add_argument('--n_epochs', '-e', type=int, default=1)
    parser.add_argument('--sigma', '-s', type=float, default=0.15)
    parser.add_argument('--lr', '-r', type=float, default=1e-4)
    parser.add_argument('--w_len', '-w', type=int, default=64)
    parser.add_argument('--zt', type=float, nargs='+')
    parser.add_argument('--hold_one_out', type=int, default=None)
    parser.add_argument('--rand_heldouts', action='store_true')

    ### Inference Args ###
    parser.add_argument('--n_infer', '-i', type=int, default=1000)
    parser.add_argument('--t_infer', '-t', type=int, default=400)

    ### Evaluation Args ###
    parser.add_argument('--eval_method', type=str, default='nearest',
                        choices=['nearest', 'interp'])
    parser.add_argument('--eval_zt_idx', type=int, nargs='+')
    parser.add_argument('--reg', type=float, default=0.1)

    ### Plotting Args ###
    parser.add_argument('--plot_d1', type=int, default=0)
    parser.add_argument('--plot_d2', type=int, default=1)
    parser.add_argument('--plot_n_background', type=int, default=2000)
    parser.add_argument('--plot_n_highlight', type=int, default=15)
    parser.add_argument('--plot_n_pairs', type=int, default=20)
    parser.add_argument('--plot_n_trajs', type=int, default=5)
    parser.add_argument('--plot_n_snaps', type=int, default=11)
    parser.add_argument('--plot_interval', type=int, default=200)
    parser.add_argument('--plot_fps', type=int, default=5)

    ### WandB Args ###
    parser.add_argument('--entity', type=str, default='')
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--resume', action='store_const', const=True, default=False)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--no_wandb', action='store_const', const='disabled',
                        default='online', dest='wandb_mode')

    ### Misc Args ###
    parser.add_argument('--outdir', '-o', type=str, default=None)
    parser.add_argument('--nogpu', action='store_true')

    args = parser.parse_args()

    ### Process Args ###
    ## Train Args
    dataname = args.dataname
    scaler_type = args.scaler_type
    flowmatcher = args.flowmatcher
    agent_type = args.agent_type
    t_sampler = args.t_sampler
    diff_ref = args.diff_ref
    window_size = args.window_size
    spline = args.spline
    monotonic = args.monotonic
    modelname = args.modelname
    modeldepth = args.modeldepth
    method = args.method
    batch_size = args.batch_size
    n_steps = args.n_steps
    n_epochs = args.n_epochs
    sigma = args.sigma
    lr = args.lr
    w = args.w_len
    zt = args.zt
    hold_one_out = args.hold_one_out
    rand_heldouts = args.rand_heldouts

    ## Inference Args
    n_infer = args.n_infer
    t_infer = args.t_infer

    ## Eval Args
    eval_method = args.eval_method
    eval_zt_idx = args.eval_zt_idx
    reg = args.reg

    ## Plot Args
    plot_d1 = args.plot_d1
    plot_d2 = args.plot_d2
    plot_n_background = args.plot_n_background
    plot_n_highlight = args.plot_n_highlight
    plot_n_pairs = args.plot_n_pairs
    plot_n_trajs = args.plot_n_trajs
    plot_n_snaps = args.plot_n_snaps
    plot_interval = args.plot_interval
    plot_fps = args.plot_fps

    ## WandB Args
    entity = args.entity
    project = args.project
    run_name = args.run_name
    group = args.group
    resume = args.resume
    log_interval = args.log_interval
    wandb_mode = args.wandb_mode

    if wandb_mode == 'online' and resume:
        run_id = get_run_id(entity, project, run_name)
    else:
        run_id = None

    ## Set Up CUDA/CPU
    device = get_device(args.nogpu)

    #### Set Up Experiment
    ### Load Data
    res = load_data_and_marginals(dataname)
    data, testdata, datalabels, testdatalabels, margtimes, marginals = res
    if marginals is not None:
        data = [data[marg] for marg in marginals]
        testdata = [testdata[marg] for marg in marginals]

    ### Scale and Set Timepoints
    zt = build_zt(zt, marginals)
    if eval_zt_idx is not None:
        eval_zt_idx = np.array(eval_zt_idx)

    args.zt = zt
    args.eval_zt_idx = eval_zt_idx
    if eval_zt_idx is not None \
        and not (np.all(eval_zt_idx >= 0) \
        and np.all(eval_zt_idx < zt.shape[0])):
        raise ValueError('Some eval_zt_idx values are not in the range [0, zt.shape[0])')

    ### Set Flags and Finish Experiment Setup
    dim = data[0].shape[1]
    is_sb = flowmatcher == 'sb'
    outdir = set_up_exp(args)

    #### Pre-processing
    ### Hold out timepoints if specified
    zt_rem_idxs = np.arange(zt.shape[0], dtype=int)
    if hold_one_out is not None:
        # list of timepoints after holding out
        zt_rem_idxs = np.delete(zt_rem_idxs, hold_one_out)
    data = [data[i] for i in zt_rem_idxs]
    if datalabels is not None:
        datalabels = [datalabels[i] for i in zt_rem_idxs]

    ### Normalize Data
    if scaler_type == 'minmax':
        scaler = IndexableMinMaxScaler()
    elif scaler_type == 'standard':
        scaler = IndexableStandardScaler()
    elif scaler_type == 'noop':
        scaler = IndexableNoOpScaler()
    else:
        raise ValueError(f'Somehow got invalid scaler_type {scaler_type}.')
    normdata = scaler.fit_transform(data)
    norm_x0 = scaler.transform(testdata)[0]

    ### Set Up Agent
    if agent_type == 'mm':  # currently not implemented
        extra_args = (window_size)
    else:
        extra_args = ()
    agent = build_agent(
        agent_type,
        args,
        zt_rem_idxs,
        dim,
        *extra_args,
        device=device
    )

    #### Run Experiment
    ### Create wandb run and set it to agent
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        config=args,  # type: ignore
        mode=wandb_mode,
        id=run_id,
        name=run_name,
        resume='allow'
    )
    agent.set_run(run)

    ### Training
    print(f'Using Agent: {agent.__class__.__name__}')

    n = n_epochs * n_steps
    flow_losses = np.zeros((n_epochs, n_steps))
    ode_eval_losses_dict = {}

    if is_sb:
        score_losses = np.zeros((n_epochs, n_steps))
        sde_eval_losses_dict = {}
    else:
        score_losses = None
        sde_eval_losses_dict = None

    pbar = tqdm(total=n)
    for i in range(n_epochs):
        do_marginals_at_epoch_log = (i + 1) % log_interval == 0 \
                                    or i == 0 \
                                    or (i + 1) == n_epochs
        flow_losses_i, score_losses_i = agent.train(normdata, pbar=pbar)
        flow_losses[i] = flow_losses_i
        if is_sb:
            score_losses[i] = score_losses_i  # type: ignore

        ### Inference (Trajectory Generation)
        ode_traj, sde_traj = agent.traj_gen(norm_x0)

        ### Rescale Trajs to Original Domain
        ode_traj = scaler.inverse_transform(ode_traj)
        ode_traj_at_zt = agent._get_traj_at_zt(ode_traj, zt)
        np.save(f'{outdir}/ode_traj_epoch{i+1}.npy', ode_traj)
        np.save(f'{outdir}/ode_traj_at_zt_epoch{i+1}.npy', ode_traj_at_zt)
        ode_eval_losses_dict_i = agent.traj_eval(testdata, ode_traj, zt)

        ## log current ode_eval_logs_dict
        print('Logging evaluation metrics over epochs at all marginals for ODE trajs...')
        for metricname, evals in ode_eval_losses_dict_i.items():
            ## save metric over epochs for individual marginals
            for m in range(evals.shape[0]):
                run.log(
                    {
                        f'{metricname}/ode_epochs_at_marginal_{m+1:0>2d}' : evals[m],
                    },
                    commit=False
                )
            ## save metric over marginals for individual epochs
            ## only do so on certain epochs to not flood wandb
            ## TODO maybe fix later to see if we can log everything
            ##      and just display desired epochs on wandb?
            ##      If this is updated, remember to do the same for
            ##      the sde trajs
            if not do_marginals_at_epoch_log:
                ## flush logs from epochs_at_marginal to wandb
                run.log({})
            else:
                ## otherwise, log marginals_at_epoch
                print(f'Logging {metricname} over marginals at epoch {i+1} for ODE trajs...')
                for m in range(evals.shape[0]):
                    run.log(
                        {
                            f'{metricname}/ode_marginals_at_epoch_{i+1:0>3d}' : evals[m],
                        }
                    )

        update_eval_losses_dict(
            ode_eval_losses_dict, ode_eval_losses_dict_i
        )

        if is_sb:
            sde_traj = scaler.inverse_transform(sde_traj)
            sde_traj_at_zt = agent._get_traj_at_zt(sde_traj, zt)
            np.save(f'{outdir}/sde_traj_epoch{i+1}.npy', sde_traj)
            np.save(f'{outdir}/sde_traj_at_zt_epoch{i+1}.npy', sde_traj_at_zt)
            sde_eval_losses_dict_i = agent.traj_eval(testdata, sde_traj, zt)

            ## log current sde_eval_logs_dict
            print('Logging evaluation metrics over epochs at all marginals for SDE trajs...')
            for metricname, evals in sde_eval_losses_dict_i.items():
                ## save metric over epochs for individual marginals
                for m in range(evals.shape[0]):
                    run.log(
                        {
                            f'{metricname}/sde_epochs_at_marginal_{m+1:0>2d}' : evals[m],
                        },
                        commit=False
                    )
                ## flush logs from epochs_at_marginal to wandb
                ## and corresponding epoch for x-axis
                run.log({'epochs' : i+1})
                ## save metric over marginals for individual epochs

                if do_marginals_at_epoch_log:
                    ## log marginals_at_epoch
                    print(f'Logging {metricname} over marginals at epoch {i+1} for SDE trajs...')
                    for m in range(evals.shape[0]):
                        run.log(
                            {
                                f'{metricname}/sde_marginals_at_epoch_{i+1:0>3d}' : evals[m],
                                'marginals' : m
                            }
                        )

            update_eval_losses_dict(
                sde_eval_losses_dict, sde_eval_losses_dict_i
            )
    pbar.close()

    #### Save Models
    agent.save_models(outdir)

    #### Save Losses
    np.save(f'{outdir}/flow_losses.npy', flow_losses)
    if is_sb:
        np.save(f'{outdir}/score_losses.npy', score_losses)  # type: ignore

    #### Save Evaluations
    write_eval_losses(ode_eval_losses_dict, outdir, is_sb=False)
    if is_sb:
        write_eval_losses(sde_eval_losses_dict, outdir, is_sb=True)

    #### Report heldout evals if eval_zt_idx is specified for convenience
    if eval_zt_idx is not None:
        print()
        report_heldout_evals(
            ode_eval_losses_dict, eval_zt_idx, outdir, is_sb=False
        )

        if is_sb:
            report_heldout_evals(
                sde_eval_losses_dict, eval_zt_idx, outdir, is_sb=True
            )

    #### Plotting
    res_plotter = Plotter(run, args, outdir, d1=plot_d1, d2=plot_d2)
    print('Plotting results...')

    res_plotter.plot_all(
        testdata, ode_traj,  # type: ignore
        flow_losses, ode_eval_losses_dict,
        legend=True, score=False
    )

    if is_sb:
        res_plotter.plot_all(
            testdata, sde_traj,  # type: ignore
            score_losses, sde_eval_losses_dict,
            legend=True, score=True
        )

    run.finish()


if __name__ == '__main__':
    main()
