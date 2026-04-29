### We write the CP(N) action in the language of O(N) model.

import torch

from .util import apply_omega

class CPNQuarticTheory:
    """
    Quartic CP(N) action assuming samples in the real representation.

    Real vector representation x is assumed to be ordered as:
      (Re[x0], Re[x1], ..., Im[x0], Im[x1], ...).
    """
    def __init__(self, beta):
        self.beta = beta

    def energy(self, x):
        """Compute action of x with shape (bs, 2*Nc, L1, L2, ...)"""
        assert x.shape[1] % 2 == 0, 'x must be shape (bs, 2*Nc, L1, L2, ...)'
        Nd = len(x.shape)-2
        # hack so that vmap behaves nicely
        S = torch.zeros_like(x.flatten(1)[:,0])
        inds = tuple(range(1, Nd+1))
        for mu in range(Nd):
            x_fwd = torch.roll(x, -1, dims=2+mu)
            h1 = (x * x_fwd).sum(1)**2
            S = S - h1.sum(inds)
            x_fwd_omega = apply_omega(x_fwd, dim=1)
            h2 = (x * x_fwd_omega).sum(1)**2
            S = S - h2.sum(inds)
        return S

    def action(self, x):
        return self.beta * self.energy(x)

    def gradE_auto(self, x):
        f = lambda x: self.action(x[None])[0]
        gx = torch.func.vmap(torch.func.grad(f))(x)
        gx = gx - x*(x*gx).sum(1, keepdim=True)
        return gx

    def gradE(self, x):
        assert x.shape[1] % 2 == 0, 'x must be shape (bs, 2*Nc, L1, L2, ...)'
        Nd = len(x.shape)-2
        gx = torch.zeros_like(x)
        for mu in range(Nd):
            # foward
            xp = torch.roll(x, -1, dims=2+mu)
            gx = gx - 2*xp*(x*xp).sum(1, keepdim=True)
            # omega * forward
            xp = apply_omega(xp, dim=1)
            gx = gx - 2*xp*(x*xp).sum(1, keepdim=True)
            # backward
            xp = torch.roll(x, 1, dims=2+mu)
            gx = gx - 2*xp*(x*xp).sum(1, keepdim=True)
            # omega * backward
            xp = apply_omega(xp, dim=1)
            gx = gx - 2*xp*(x*xp).sum(1, keepdim=True)
        gx = gx - x*(x*gx).sum(1, keepdim=True)
        return gx

    def grad_action(self, x):
        return self.beta * self.gradE(x)

def _test_grad_cpn_action():
    from .util import sample_hot, torch_dtype
    with torch_dtype(torch.double):
        action = CPNQuarticTheory(1.0)
        x = sample_hot(5, (4, 4), n=4)
        g1 = action.gradE(x)
        g2 = action.gradE_auto(x)
    assert torch.allclose(g1, g2)
    print('[PASSED test_grad_cpn_action]')

if __name__ == '__main__':
    _test_grad_cpn_action()
