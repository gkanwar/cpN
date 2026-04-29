import contextlib
import numpy as np
import torch

# Global configuration
ND = 2

def grab(x):
    return x.detach().cpu().numpy()

def flatten_jac(jac):
    assert len(jac.shape) % 2 == 1
    l = len(jac.shape)//2
    return jac.flatten(1,l).flatten(2,1+l)

def expand_n(x, n):
    """Unsqueeze the rightmost `n` dimensions of `x`"""
    inds = (slice(None),)*len(x.shape) + (None,)*n
    return x[inds]

def bcast_to_latt(w, *, Nd=ND):
    """Broadcast batched scalar to lattice shape"""
    w = torch.as_tensor(w)
    if len(w.shape) == 0:
        w = w[None]
    assert len(w.shape) == 1
    return expand_n(w, Nd)

def bcast_to_spins(w, *, Nd=ND):
    """Broadcast batched scalar to spins (lattice + color) shape"""
    w = torch.as_tensor(w)
    if len(w.shape) == 0:
        w = w[None]
    assert len(w.shape) == 1
    return expand_n(w, Nd+1)

def compute_ess(logw):
    return np.exp(2*np.logaddexp.reduce(logw) - np.logaddexp.reduce(2*logw)) / len(logw)

def project(p, x, *, dim):
    """Project p onto tangent space of the sphere at x."""
    return p - x * (p * x).sum(dim, keepdim=True)

def rotate(x, dx, *, dim):
    """
    Rotate (x,dx) -> (x', dx') assuming dx lives in tangent space at x.
    The resulting dx' lives in the tangent space of x' and the update is
    reversible (x', -dx') -> (x, -dx).
    """
    alpha_sq = (dx*dx).sum(dim, keepdim=True)
    EPS = 1e-8
    # NOTE(gkanwar): this is needed to avoid propagating NaNs in wrong branch
    alpha = alpha_sq.clamp(min=0.1*EPS).sqrt()
    cos_a = torch.where(alpha_sq <= EPS, 1 - alpha_sq/2 + alpha_sq**2/24, torch.cos(alpha))
    a_sin_a = torch.where(alpha_sq <= EPS, alpha_sq - alpha_sq**2/6, alpha*torch.sin(alpha))
    sinc_a = torch.where(alpha_sq <= EPS, 1 - alpha_sq/6 + alpha_sq**2/120, torch.sinc(alpha/np.pi))
    xp = cos_a*x + sinc_a*dx
    dxp = cos_a*dx - a_sin_a*x
    return xp, dxp

def apply_omega(x, *, dim):
    """Given a real representation x of a CP(N) variable, apply Omega matrix."""
    if dim < 0:
        dim += len(x.shape)
    assert x.shape[dim] % 2 == 0, 'x must have shape (2*Nc) along dim'
    Nc = x.shape[dim] // 2
    ind1 = (slice(None),)*dim + (slice(0,Nc),)
    ind2 = (slice(None),)*dim + (slice(Nc,None),)
    x_re, x_im = x[ind1], x[ind2]
    return torch.cat([-x_im, x_re], dim=dim)

def sample_hot(batch_size, shape, n):
    """
    Sample uniform samples on S^{n-1} to initialize either O(N) or CP(N).
    """
    eta = torch.randn((batch_size,n)+shape)
    eta /= (eta**2).sum(1, keepdim=True).sqrt()
    return eta

@contextlib.contextmanager
def torch_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
