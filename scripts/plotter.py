import numpy as np
from matplotlib import (
    pyplot as plt,
    colors as mcolors,
    animation,
)
from scipy.stats import scoreatpercentile
import scprep
import wandb


class Plotter():
    def __init__(self, run, args, outdir, d1=0, d2=1):
        self.run = run
        self.zt = args.zt
        self.test_n = 200  ## number of samples from testdata
        self.ntrajs = args.plot_n_trajs  ## num of trajs to highlight
        self.npairs = args.plot_n_pairs  ## num of x_0, x_M pairs to show
        self.nbackground = args.plot_n_background  ## num of background trajs to show. Should be < self.nhighlight
        self.nhighlight = args.plot_n_highlight  ## num of trajs to highlight
        self.nsnaps = args.plot_n_snaps
        self.interval = args.plot_interval
        self.fps = args.plot_fps

        self.outdir = outdir
        ## Sort colors by hue, saturation, value and name
        self.cnames = sorted(
            mcolors.TABLEAU_COLORS,
            key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c)))
        )

        ## data dims to plot
        self.d1 = d1  # x dim
        self.d2 = d2  # y dim

        ## updated when calling self._compute_xy_lims()
        self.xlim = None
        self.ylim = None

    def _plot_data(self, data, ax):
        for i in range(len(data)):
            label = f't = {self.zt[i]:.2f}'
            ax.scatter(
                data[i][:self.test_n, self.d1], data[i][:self.test_n, self.d2],
                color=self.cnames[i], label=label, marker='.', alpha=1
            )

    def _compute_xy_lims(self, data):
        ## calculate (xmin, xmax), (ymin, ymax) for consistent scale on data
        ## do not account for traj because bad traj should fly off screen
        datamin = np.nanmin(
            np.array(
                [np.nanmin(datamarg, axis=0) for datamarg in data]
            ),
            axis=0
        )
        datamax = np.nanmax(
            np.array(
                [np.nanmax(datamarg, axis=0) for datamarg in data]
            ),
            axis=0
        )
        xmin, ymin = datamin[[self.d1, self.d2]]
        xmax, ymax = datamax[[self.d1, self.d2]]

        ## calculate xy margins for prettier plots
        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05

        ## get xlim, ylim
        self.xlim = (xmin - xmargin, xmax + xmargin)
        self.ylim = (ymin - ymargin, ymax + ymargin)

    def plot_loss(self, loss, title, score=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_title(title)
        ax.plot(loss.flatten())
        ax.set_xlabel('Gradient Steps')
        ax.set_ylabel('Loss')

        if not score:  # plot flow loss
            fig.savefig(f'{self.outdir}/flow_loss.png')
            fig.savefig(f'{self.outdir}/flow_loss.pdf')
        else:  # plot score loss
            fig.savefig(f'{self.outdir}/score_loss.png')
            fig.savefig(f'{self.outdir}/score_loss.pdf')

        if not score:
            img = wandb.Image(
                data_or_path=f'{self.outdir}/flow_loss.png',
                mode='RGB'
            )
            self.run.log(
                {
                    f'evalplots/flow_loss' : img
                }
            )
        else:
            img = wandb.Image(
                data_or_path=f'{self.outdir}/score_loss.png',
                mode='RGB'
            )
            self.run.log(
                {
                    f'evalplots/score_loss' : img
                }
            )

        plt.close(fig)

    def plot_loss_epoch_means(self, loss, title, score=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_title(title)
        ax.plot(loss.mean(axis=1))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        if not score:  # plot flow loss
            fig.savefig(f'{self.outdir}/flow_loss_epochs.png')
            fig.savefig(f'{self.outdir}/flow_loss_epochs.pdf')
        else:  # plot score loss
            fig.savefig(f'{self.outdir}/score_loss_epochs.png')
            fig.savefig(f'{self.outdir}/score_loss_epochs.pdf')

        if not score:
            img = wandb.Image(
                data_or_path=f'{self.outdir}/flow_loss_epochs.png',
                mode='RGB'
            )
            self.run.log(
                {
                    f'evalplots/flow_loss_epochs' : img
                }
            )
        else:
            img = wandb.Image(
                data_or_path=f'{self.outdir}/score_loss_epochs.png',
                mode='RGB'
            )
            self.run.log(
                {
                    f'evalplots/score_loss_epochs' : img
                }
            )

        plt.close(fig)

    def plot_traj_loss_overlap(
            self, eval_losses_dict, title, legend=True, score=False
    ):
        ## plot eval_losses all on same plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_title(title)
        ax.set_xlabel('Marginals')
        ax.set_ylabel('Loss')
        for loss_name, losses in eval_losses_dict.items():
            ax.plot(losses[-1], label=loss_name)

        fig.tight_layout()

        if legend:
            leg = ax.legend(loc='best')
            for lh in leg.legend_handles:
                lh.set_alpha(1)  # type: ignore

        # plot flow loss
        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/traj_loss_overlap_{diffeq}'
        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'evalplots/traj_loss_overlap_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_traj_loss_indiv(
            self, eval_losses_dict, title, legend=True, score=False
    ):
        ## plot eval losses on individual plots
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(3*8 + 2, 2*8 + 1))
        fig.suptitle(title, fontsize=24)

        for k, (loss_name, losses) in enumerate(eval_losses_dict.items()):
            i, j = divmod(k, 2)
            ax = axs[j, i]  ## use j, i instead of i, j to iterate over columns from left to right
            ax.set_title(loss_name, fontsize=20)
            ax.set_xlabel('Marginals')
            ax.set_ylabel('Loss')
            ax.plot(losses[-1])
            _txt = f'Total Loss = {losses[-1].sum():.2f}\n' \
                   + f'Avg Loss = {losses[-1].mean():.2f}'
            ax.text(
                0.5, 0.1, _txt, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha='center', fontsize=16
            )

        fig.tight_layout()

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/traj_loss_indiv_{diffeq}'
        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'evalplots/traj_loss_indiv_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_traj_loss_epochs(
            self, eval_losses_dict, M, title, legend=True, score=False
    ):
        for m in range(M):
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(3*8 + 2, 2*8 + 1))
            fig.suptitle(title, fontsize=24)

            for k, (loss_name, losses) in enumerate(eval_losses_dict.items()):
                i, j = divmod(k, 2)
                ## use j, i instead of i, j to iterate cols from left to right
                ax = axs[j, i]
                ax.set_title(loss_name, fontsize=20)
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                n_epochs = losses.shape[0]
                xticks = np.arange(1, n_epochs+1).astype(int)
                ax.set_xticks(xticks)
                ax.plot(xticks, losses[:, m])

            fig.tight_layout()

            diffeq = 'sde' if score else 'ode'
            filename = f'{self.outdir}/traj_loss_epochs_{diffeq}_t{m}'
            fig.savefig(f'{filename}.png')
            fig.savefig(f'{filename}.pdf')

            img = wandb.Image(
                data_or_path=f'{filename}.png',
                mode='RGB'
            )
            self.run.log(
                {
                    f'evalplots/traj_loss_epochs_{diffeq}_t{m}' : img
                }
            )

            plt.close(fig)

    def plot_trajs(self, data, traj, title=None, legend=True, score=False):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        if title:
            ax.set_title(title)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        ## plot underlying test data
        self._plot_data(data, ax)

        ## plot trajs
        ax.plot(
            traj[:, :self.nbackground, self.d1],
            traj[:, :self.nbackground, self.d2],
            alpha=0.05, c='tab:olive'
        )

        ## highlight a few trajs
        for i in range(self.nhighlight - 1):
            ax.plot(
                traj[:, i, self.d1],
                traj[:, i, self.d2],
                alpha=0.9, c='k'
            )
        ax.plot(
            traj[:, self.nhighlight-1, self.d1],
            traj[:, self.nhighlight-1, self.d2],
            alpha=0.9, c='k', label='Traj'
        )
        if legend:
            leg = ax.legend(loc='best')
            for lh in leg.legend_handles:
                lh.set_alpha(1)  # type: ignore

        fig.tight_layout()

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/trajs_{diffeq}'
        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'visualizations/trajs_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_trajs_gif(self, data, traj, title=None, legend=True, score=False):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        if title:
            fig.suptitle(title, fontsize=24)

        snap_times = np.linspace(0, 1, traj.shape[0])
        snap_times_idxs = (
            np.linspace(0, traj.shape[0]-1, self.nsnaps)
            .round().astype(int)
        )

        def animate(i):
            t_idx = snap_times_idxs[i]
            t = snap_times[t_idx]
            ax.clear()
            ax.text(
                0.5, 0.1, f't = {t:.2f}',
                bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha='center', fontsize=16
            )
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            self._plot_data(data, ax)

            ## plot trajs so far
            ax.plot(
                traj[:t_idx, :self.nbackground, self.d1],
                traj[:t_idx, :self.nbackground, self.d2],
                alpha=0.05, c='tab:olive'
            )

            ## highlight a few trajs
            ax.plot(
                traj[:t_idx, :self.nhighlight-1, self.d1],
                traj[:t_idx, :self.nhighlight-1, self.d2],
                alpha=0.9, c='k'
            )
            ax.plot(
                traj[:t_idx, self.nhighlight-1, self.d1],
                traj[:t_idx, self.nhighlight-1, self.d2],
                alpha=0.9, c='k', label='Traj'
            )

            if legend:
                leg = ax.legend(loc='upper left')
                for lh in leg.legend_handles:
                    lh.set_alpha(1)  # type: ignore

        fig.tight_layout()
        ani = animation.FuncAnimation(
            fig=fig, func=animate,  # type: ignore
            frames=self.nsnaps, interval=self.interval
        )

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/trajs_{diffeq}.gif'
        ani.save(filename=filename, writer='imagemagick', fps=self.fps)

        gif = wandb.Video(
            data_or_path=filename,
            format='gif'
        )
        self.run.log(
            {
                f'visualizations/trajs_{diffeq}_gif' : gif
            }
        )

        plt.close(fig)

    def plot_traj_from_samples(
            self, data, traj, title=None, legend=True, score=False
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if title:
            ax.set_title(title)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        self._plot_data(data, ax)

        # plot start (blue) and end (red) pairs
        ax.scatter(
            traj[0, :self.npairs, self.d1], traj[0, :self.npairs, self.d2],
            s=20, alpha=1, marker='d', c="blue", zorder=3, label=r'$x_0$'
        )
        ax.scatter(
            traj[-1, :self.npairs, self.d1], traj[-1, :self.npairs, self.d2],
            s=20, alpha=1, marker='d', c='red', zorder=3, label=r'$x_1$'
        )

        for i in range(self.ntrajs-1):
            ax.plot(
                traj[:, i, self.d1], traj[:, i, self.d2],
                alpha=0.9, c="black", zorder=2
            )
        ax.plot(
            traj[:, self.ntrajs-1, self.d1], traj[:, self.ntrajs-1, self.d2],
            alpha=0.9, c="black", zorder=2, label='Traj'
        )

        if legend:
            leg = ax.legend(loc='best')
            for lh in leg.legend_handles:
                lh.set_alpha(1)

        fig.tight_layout()

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/traj_from_samples_{diffeq}'

        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'visualizations/traj_from_samples_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_traj_with_timepoints(
            self, data, traj, title=None, legend=True, score=False
    ):
        fig, axs = plt.subplots(
            nrows=self.ntrajs, ncols=self.nsnaps,
            figsize=(6*self.nsnaps, 6*self.ntrajs)
        )
        if title:
            fig.suptitle(title, fontsize=32)

        snap_times = np.linspace(0, 1, traj.shape[0])
        snap_times_idxs = (
            np.linspace(0, traj.shape[0]-1, self.nsnaps)
            .round()
            .astype(int)
        )

        for i in range(self.ntrajs):
            for j, t_idx in enumerate(snap_times_idxs):
                ax = axs[i, j]
                t = snap_times[t_idx]
                ax.set_title(f'Traj {i+1}, t = {t:.2f}')
                ax.set_xlim(self.xlim)
                ax.set_ylim(self.ylim)
                ax.set_xticks([])
                ax.set_yticks([])

                ## plot underlying test data
                self._plot_data(data, ax)
                ## plot traj so far
                ax.plot(
                    traj[:t_idx, i, self.d1], traj[:t_idx, i, self.d2],
                    alpha=0.9, c='black', label='Traj'
                )

                if legend:
                    leg = ax.legend(loc='best')
                    for lh in leg.legend_handles:
                        lh.set_alpha(1)

        fig.tight_layout()

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/traj_timepoints_{diffeq}'
        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'visualizations/traj_timepoints_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_snapshots(self, data, traj, title=None, legend=True, score=False):
        fig, axs = plt.subplots(ncols=self.nsnaps, figsize=(6*self.nsnaps, 6))
        if title:
            fig.suptitle(title, fontsize=32)

        snap_times = np.linspace(0, 1, traj.shape[0])
        snap_times_idxs = (
            np.linspace(0, traj.shape[0]-1, self.nsnaps)
            .round()
            .astype(int)
        )

        for i, t_idx in enumerate(snap_times_idxs):
            ax = axs[i]
            t = snap_times[t_idx]
            ax.set_title(f't = {t:.2f}')
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            self._plot_data(data, ax)
            ax.scatter(
                traj[t_idx, :, self.d1], traj[t_idx, :, self.d2],
                marker='.', label=r'$p_t$'
            )

            if legend:
                leg = ax.legend(loc='best')
                for lh in leg.legend_handles:
                    lh.set_alpha(1)

        fig.tight_layout()

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/snapshots_{diffeq}'
        fig.savefig(f'{filename}.png')
        fig.savefig(f'{filename}.pdf')

        img = wandb.Image(
            data_or_path=f'{filename}.png',
            mode='RGB'
        )
        self.run.log(
            {
                f'visualizations/snapshots_{diffeq}' : img
            }
        )

        plt.close(fig)

    def plot_snapshots_gif(
            self, data, traj, title=None, legend=True, score=False
    ):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        if title:
            fig.suptitle(title, fontsize=24)

        snap_times = np.linspace(0, 1, traj.shape[0])
        snap_times_idxs = (
            np.linspace(0, traj.shape[0]-1, self.nsnaps)
            .round()
            .astype(int)
        )

        def animate(i):
            t_idx = snap_times_idxs[i]
            t = snap_times[t_idx]
            ax.clear()
            ax.text(
                0.5, 0.1, f't = {t:.2f}',
                bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha='center', fontsize=16
            )
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            self._plot_data(data, ax)

            ## plot current position
            ax.scatter(
                traj[t_idx, :, self.d1], traj[t_idx, :, self.d2],
                marker='.', c='tab:blue', label=r'$p_t$'
            )

            if legend:
                leg = ax.legend(loc='upper left')
                for lh in leg.legend_handles:
                    lh.set_alpha(1)  # type: ignore

        fig.tight_layout()

        ani = animation.FuncAnimation(
            fig=fig, func=animate,  # type: ignore
            frames=self.nsnaps, interval=self.interval
        )

        diffeq = 'sde' if score else 'ode'
        filename = f'{self.outdir}/snapshots_{diffeq}.gif'
        ani.save(filename=filename, writer='imagemagick', fps=self.fps)

        gif = wandb.Video(
            data_or_path=filename,
            format='gif'
        )
        self.run.log(
            {
                f'visualizations/snapshots_{diffeq}_gif' : gif
            }
        )

        plt.close(fig)

    def plot_all(
            self, data, traj, loss, eval_losses_dict, legend=True, score=False
    ):
        loss_title = 'Score Loss' if score else 'Flow Loss'
        titlebase = 'SDE' if score else 'ODE'
        traj_loss_title = f'{titlebase} Eval Loss'

        ## Loss Plots
        self.plot_loss(loss, loss_title, score=score)
        self.plot_loss_epoch_means(loss, loss_title, score=score)
        self.plot_traj_loss_overlap(
            eval_losses_dict, traj_loss_title, score=score
        )
        self.plot_traj_loss_indiv(
            eval_losses_dict, traj_loss_title, score=score
        )
        self.plot_traj_loss_epochs(
            eval_losses_dict, len(data), traj_loss_title, score=score
        )

        ## Traj Plots
        self._compute_xy_lims(data)
        self.plot_trajs(
            data, traj, title=f'{titlebase} Trajectories',
            legend=legend, score=score
        )
        self.plot_trajs_gif(
            data, traj, title=f'{titlebase} Trajectories',
            legend=legend, score=score
        )
        self.plot_traj_from_samples(
            data, traj, title=f'{titlebase} Sample Trajectories',
            legend=legend, score=score
        )
        self.plot_traj_with_timepoints(
            data, traj, title=f'{titlebase} Cumulative Trajectories with Timepoints',
            legend=legend, score=score
        )
        self.plot_snapshots(
            data, traj, title=f'{titlebase} Snapshots', legend=legend, score=score
        )
        self.plot_snapshots_gif(
            data, traj, title=f'{titlebase} Snapshots', legend=legend, score=score
        )

