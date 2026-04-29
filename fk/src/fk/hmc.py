import torch
import tqdm.auto as tqdm

from .util import rotate, project, expand_n, grab

def leapfrog(x, p, *, n_leap, dtau, force):
    """Leapfrog on the O(N) manifold with geodesic updates."""

    def _rotate(x, p, dtau):
        x, pp = rotate(x, p*dtau, dim=1)
        return x, pp/dtau

    p = p + (dtau/2) * force(x)
    for _ in range(n_leap - 1):
        x, p = _rotate(x, p, dtau)
        p = p + dtau * force(x)
    x, p = _rotate(x, p, dtau)
    p = p + (dtau/2) * force(x)
    return x, p

def hmc_traj(x, *, theory, n_leap, dtau, ar=True):
    # tangent-space momenta
    p = torch.randn_like(x)
    p = project(p, x, dim=1)

    force = lambda x: -theory.grad_action(x)

    if not ar:
        return leapfrog(x, p, n_leap=n_leap, dtau=dtau, force=force)[0]

    K = 0.5 * (p**2).flatten(1).sum(1)
    S = theory.action(x)

    xp, pp = leapfrog(x, p, n_leap=n_leap, dtau=dtau, force=force)

    Kp = 0.5 * (pp**2).flatten(1).sum(1)
    Sp = theory.action(xp)

    dH = Sp + Kp - S - K
    acc = torch.rand(dH.shape) < torch.exp((-dH).clamp(max=0.0))
    
    x = torch.where(expand_n(acc, len(x.shape)-1), xp, x)
    return dict(x=x, acc=acc, dH=dH)

def run_hmc(x0, *, theory, n_leap, dtau, n_iter, n_therm=0, progress=False):
    """Run batched HMC"""
    x = x0.clone()
    acc_tot = 0

    it = range(n_therm)
    if progress:
        it = tqdm.tqdm(it, leave=False)
    for _ in it:
        res = hmc_traj(x, theory=theory, n_leap=n_leap, dtau=dtau)
        x = res['x']

    ens = []
    it = range(n_iter)
    if progress:
        it = tqdm.tqdm(it, leave=False)
    for _ in it:
        res = hmc_traj(x, theory=theory, n_leap=n_leap, dtau=dtau)
        ens.append(res['x'].clone())
        acc_tot += grab(res['acc'].mean())

    ens = torch.stack(ens, dim=0)
    acc_rate  = acc_tot / n_iter
    return ens, acc_rate

def _test_hmc():
    from .util import torch_dtype, sample_hot
    from .theory import CPNQuarticTheory
    torch.manual_seed(1234)
    with torch_dtype(torch.double):
        theory = CPNQuarticTheory(0.5)
        x = sample_hot(5, (4, 4), n=4)
        res = hmc_traj(x, theory=theory, n_leap=10, dtau=0.001)
    assert torch.all(res['dH'].abs() < 1e-6)
    print('[PASSED test_hmc]')
if __name__ == '__main__':
    _test_hmc()
