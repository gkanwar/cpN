### Feynman-Kac estimates for given input CP(N) ensemble, accumulating stats
### incrementally so that we can in principle converge to precise results.

import argparse
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
import threading
import torch
import tqdm.auto as tqdm

from fk.theory import CPNQuarticTheory
from fk.langevin import run_fk
from fk.util import grab, wrap, project_cpn_u1

if torch.cuda.is_available():
    torch.set_default_device('cuda')

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

    plt.ion()
    idx = args.n_skip
    fig, axes = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
    axes[0].set_title('Config')
    axes[1].set_title('Mean b')
    axes[2].set_title('Var b')
    data_cfg = axes[0].imshow(np.zeros_like(ens[idx,0]), vmin=-np.pi, vmax=np.pi, cmap='twilight')
    def _plot_cfg():
        th1 = np.arctan2(ens[idx,1], ens[idx,0])
        th2 = np.arctan2(ens[idx,3], ens[idx,2])
        data_cfg.set_data(wrap(th1 - th2))
        fig.canvas.draw_idle()
    data_bx = axes[1].imshow(np.zeros_like(ens[idx,0]), cmap='RdBu')
    data_var_bx = axes[2].imshow(np.zeros_like(ens[idx,0]))
    cb1 = fig.colorbar(data_bx, ax=axes[1])
    cb2 = fig.colorbar(data_var_bx, ax=axes[2])
    def _plot_f(bx_mean, bx_var, n):
        data_bx.set_data(grab(bx_mean)[0,0])
        data_var_bx.set_data(grab(bx_var)[0,0])
        vmax = np.max(np.abs(grab(bx_mean)[0,0]))
        data_bx.set_clim(vmin=-vmax, vmax=vmax)
        data_var_bx.autoscale()
        cb1.update_normal(data_bx)
        cb2.update_normal(data_var_bx)
        fig.canvas.draw_idle()
        plt.pause(0.05)
    plt.pause(0.05)

    # evaluate everything!
    bx, var_bx = [], []
    inds = torch.arange(args.n_skip, len(ens))
    for chunk in tqdm.tqdm(torch.split(inds, args.batch_size)):
        idx = grab(chunk[0])
        _plot_cfg()
        x = torch.as_tensor(ens[grab(chunk)]).to(torch.get_default_dtype())
        proj = lambda b: project_cpn_u1(b, x, dim=1)
        res = run_fk(x, plot_f=_plot_f, proj=proj, **kwargs)
        bx.append(res['bx'])
        var_bx.append(res['var_bx'])
    bx = np.concatenate(bx)
    var_bx = np.concatenate(var_bx)

    print(f'Saving to {args.out_fname}')
    np.save(args.out_fname, dict(bx=bx, var_bx=var_bx, beta=args.beta))

if __name__ == '__main__':
    main()
