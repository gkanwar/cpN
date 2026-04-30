### Feynman-Kac estimates for given input CP(N) ensemble, accumulating stats
### incrementally so that we can in principle converge to precise results.

import argparse
import os
import matplotlib

HEADLESS = (
    os.environ.get('DISPLAY') is None and
    os.environ.get('WAYLAND_DISPLAY') is None
)
if HEADLESS:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto as tqdm

from fk.theory import CPNQuarticTheory
from fk.langevin import fk_phi
from fk.util import grab, wrap

if torch.cuda.is_available():
    torch.set_default_device('cuda')


def run_fk_phi(x0, *, theory, dSdt, n_iter, dtau, n_accum, plot_f=None):
    phi_mean = torch.zeros_like(x0.flatten(1)[:,0])
    phi_M2 = torch.zeros_like(phi_mean)

    def _update_running(phi_n, n):
        nonlocal phi_mean, phi_M2
        old_mean = phi_mean.clone()
        phi_mean += (phi_n - phi_mean) / n
        phi_M2 += (phi_n - old_mean) * (phi_n - phi_mean)
        if plot_f is not None:
            plot_f(phi_mean, phi_M2/n, n=n)

    kwargs = dict(theory=theory, dSdt=dSdt, n_iter=n_iter, dtau=dtau)
    for i in tqdm.tqdm(range(n_accum), leave=False):
        phi_i = fk_phi(x0, **kwargs)
        _update_running(phi_i, i+1)

    return dict(phi=grab(phi_mean), var_phi=grab(phi_M2/n_accum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--ens_fname', type=str, required=True)
    parser.add_argument('--n_skip', type=int, default=0, help='Num starting configs to skip')
    parser.add_argument('--fk_iter', type=int, required=True)
    parser.add_argument('--fk_tau', type=float, required=True)
    parser.add_argument('--fk_accum', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--out_fname', type=str, required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.double)

    ens = np.load(args.ens_fname)
    theory = CPNQuarticTheory(args.beta)
    dSdt = lambda x: theory.energy(x)

    n_iter = args.fk_iter
    dtau = args.fk_tau / n_iter
    n_accum = args.fk_accum
    kwargs = dict(theory=theory, dSdt=dSdt, n_iter=n_iter, dtau=dtau, n_accum=n_accum)

    plot_f = None
    def _plot_cfg():
        pass
    if not HEADLESS:
        plt.ion()
        idx = args.n_skip
        fig, axes = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
        axes[0].set_title('Config')
        axes[1].set_title('Mean phi')
        axes[2].set_title('Var phi')
        data_cfg = axes[0].imshow(np.zeros_like(ens[idx,0]), vmin=-np.pi, vmax=np.pi, cmap='twilight')
        hist_phi, hist_var_phi = [], []
        def _plot_cfg():
            hist_phi.clear()
            hist_var_phi.clear()
            Nc = ens.shape[1]//2
            th1 = np.arctan2(ens[idx,Nc], ens[idx,0])
            th2 = np.arctan2(ens[idx,Nc+1], ens[idx,1])
            data_cfg.set_data(wrap(th1 - th2))
            axes[1].cla()
            axes[2].cla()
            axes[1].set_title('Mean phi')
            axes[2].set_title('Var phi')
            fig.canvas.draw_idle()
        def plot_f(phi_mean, phi_var, n):
            phi_mean = grab(phi_mean)
            phi_var = grab(phi_var)
            hist_phi.append(phi_mean.copy())
            hist_var_phi.append(phi_var.copy())
            _hist_phi = np.stack(hist_phi)
            _hist_var_phi = np.stack(hist_var_phi)
            x = np.arange(1, n+1)
            _hist_phi_err = _hist_var_phi / np.sqrt(x)[:,None]
            axes[1].cla()
            axes[2].cla()
            axes[1].set_title('Mean phi')
            axes[2].set_title('Var phi')
            axes[1].plot(x, _hist_phi)
            if len(x) >= 4:
                inds = slice(4, None)
                for i in range(_hist_phi.shape[-1]):
                    slice_inds = (inds, i)
                    axes[1].fill_between(x[inds], (_hist_phi + _hist_phi_err)[slice_inds], (_hist_phi - _hist_phi_err)[slice_inds], ec='none', alpha=0.5)
            axes[2].plot(x, _hist_var_phi)
            axes[1].set_xlim(1, max(1, n))
            axes[2].set_xlim(1, max(1, n))
            fig.suptitle(f'Accum {n}/{n_accum}')
            fig.canvas.draw_idle()
            plt.pause(0.05)
        plt.pause(0.05)

    phi, var_phi = [], []
    inds = torch.arange(args.n_skip, len(ens))
    for chunk in tqdm.tqdm(torch.split(inds, args.batch_size)):
        idx = grab(chunk[0])
        _plot_cfg()
        x = torch.as_tensor(ens[grab(chunk)]).to(torch.get_default_dtype())
        res = run_fk_phi(x, plot_f=plot_f, **kwargs)
        phi.append(res['phi'])
        var_phi.append(res['var_phi'])
    phi = np.concatenate(phi)
    var_phi = np.concatenate(var_phi)

    print(f'Saving to {args.out_fname}')
    np.save(args.out_fname, dict(phi=phi, var_phi=var_phi, beta=args.beta))


if __name__ == '__main__':
    main()
