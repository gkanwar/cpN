import numpy as np
import torch
import tqdm.auto as tqdm

from .util import rotate, project, grab

def langevin_step(x, *, theory, dtau):
    eta = torch.randn_like(x)
    eta = project(eta, x, dim=1)
    dx = -dtau*theory.grad_action(x) + np.sqrt(2*dtau)*eta
    xp, _ = rotate(x, dx, dim=1)
    return xp

def fk_phi(x, *, theory, dSdt, n_iter, dtau, progress=False):
    """Run batched langevin without accept/reject."""
    phi = 0
    it = range(n_iter)
    if progress:
        it = tqdm.tqdm(it)
    for _ in it:
        x = langevin_step(x, theory=theory, dtau=dtau)
        phi = phi - dtau*dSdt(x)
    return phi

def _fk_single(x0, *, theory, dSdt, n_iter, dtau):
    f = lambda x: fk_phi(x[None], theory=theory, dSdt=dSdt, n_iter=n_iter, dtau=dtau)[0]
    return torch.func.vmap(torch.func.grad(f), randomness='different')(x0)

def run_fk(x0, *, theory, dSdt, n_iter, dtau, n_accum, plot_f=None, proj=None):
    bx_mean = torch.zeros_like(x0)
    bx_M2 = torch.zeros_like(x0)

    # running estimate of mean and variance using the Welford algorithm
    def _update_running(bx_n, n):
        nonlocal bx_mean, bx_M2
        old_mean = bx_mean.clone()
        bx_mean += (bx_n - bx_mean) / n
        bx_M2 += (bx_n - old_mean)*(bx_n - bx_mean)
        if plot_f is not None:
            plot_f(bx_mean, bx_M2/n, n=n)

    kwargs = dict(theory=theory, dSdt=dSdt, n_iter=n_iter, dtau=dtau)
    for i in tqdm.tqdm(range(n_accum), leave=False):
        bx_i = _fk_single(x0, **kwargs)
        if proj is not None:
            bx_i = proj(bx_i)
        _update_running(bx_i, i+1)

    return dict(bx=grab(bx_mean), var_bx=grab(bx_M2/n_accum))
