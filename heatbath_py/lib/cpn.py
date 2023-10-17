import numpy as np
import scipy as sp
import scipy.linalg
import scipy.interpolate
import scipy.special
import tqdm

# CP(N-1) field
def make_cold_cfg(shape, N):
    z = np.zeros(shape + (N,), dtype=np.complex128)
    z[...,0] = 1.0
    return z

def make_hot_cfg(shape, N, *, rng):
    shape = shape + (N,)
    z = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    normsq = np.sum(np.abs(z)**2, axis=-1, keepdims=True)
    z /= np.sqrt(normsq)
    return z

def make_topo_cfg(shape, N):
    assert len(shape) == 2, 'specialized for 2d'
    assert N == 2, 'specialized for N=2'
    Lx, Lt = shape
    assert Lt % 2 == 0, 'specialized for even Lt'
    phi = np.pi*np.arange(0, Lt//2)/Lt
    zA = np.tile(
        np.stack((np.cos(phi), np.sin(phi)), axis=-1),
        (Lx,1,1))
    th = 2*np.pi*np.arange(0, Lx)/Lx
    phi = np.pi*np.arange(0, Lt//2)/Lt
    th, phi = np.meshgrid(th, phi, indexing='ij')
    zB = np.stack((1j*np.sin(phi)*np.exp(1j*th), np.cos(phi)), axis=-1)
    assert zA.shape == zB.shape
    z = np.concatenate((zA, zB), axis=-2)
    assert z.shape == shape + (N,)
    assert np.allclose(np.real(inner(z,z)), 1.0)
    return z
    

def inner(z1, z2):
    return np.sum(np.conj(z1) * z2, axis=-1)


# SO(N) matrix sampled from gaussian about Id
def gaussian_soN_matrix(shape, N, *, sigma, rng):
    A = sigma*rng.normal(size=shape+(N,N))
    A = (A - A.swapaxes(-1,-2)) / 2
    return sp.linalg.expm(A)
def gaussian_suN_matrix(shape, N, *, sigma, rng):
    A = sigma*(rng.normal(size=shape+(N,N)) + 1j*rng.normal(size=shape+(N,N)))
    A = (A - A.swapaxes(-1,-2).conj()) / 2
    return sp.linalg.expm(A)
def propose_update(z, *, sigma, rng):
    shape, N = z.shape[:-1], z.shape[-1]
    # X = gaussian_soN_matrix(shape, 2*N, sigma=sigma, rng=rng)
    X = gaussian_suN_matrix(shape, N, sigma=sigma, rng=rng)
    # phi = np.concatenate((np.real(z), np.imag(z)), axis=-1)
    # phip = np.einsum('...ij,...j->...i', X, phi)
    # zp = phip[...,:N] + 1j*phip[...,N:]
    zp = np.einsum('...ij,...j->...i', X, z)
    return zp

# Metropolis tools
def get_checkerboard_mask(p, *,  shape):
    arrs = [np.arange(Lx) for Lx in shape]
    return sum(np.ix_(*arrs)) % 2 == p
def reunit(cfg):
    cfg /= np.sqrt(np.sum(np.abs(cfg)**2, axis=-1, keepdims=True))


class CPNQuarticAction:
    def __init__(self, beta):
        self.beta = beta
    ### S = -beta sum_<xy> (|zx^dag . zy|^2 - 1)
    ### (trivial subtraction puts the minimum action at 0)
    def action(self, z):
        latt_shape,  N = z.shape[:-1], z.shape[-1]
        S = 0
        for mu in range(len(latt_shape)):
            z_fwd = np.roll(z, -1, axis=mu)
            z_bwd = np.roll(z, 1, axis=mu)
            S += np.sum(np.abs(inner(z_fwd, z))**2 - 1)
            S += np.sum(np.abs(inner(z_bwd, z))**2 - 1)
        return -self.beta * S
    ### Sx = -beta sum_y (zx . zy^dag)(zx^dag . zy)
    ###    = -beta sum_y sum_ij zx_i zy^dag_i zx^dag_j zy_j
    ###    = -beta zx^dag . sum_y (zy zy^dag) . zx
    def local_action(self, z):
        latt_shape, N = z.shape[:-1], z.shape[-1]
        S = np.zeros(latt_shape)
        for mu in range(len(latt_shape)):
            S -= self.beta * (
                np.abs(inner(z, np.roll(z, -1, axis=mu)))**2 +
                np.abs(inner(z, np.roll(z, 1, axis=mu)))**2
            )
        return S
    ### z(omega) = exp(omega) z, omega \in so(2N)
    ### dS/d(omega_x) = -beta (zx zx^dag sum_y (zy zy^dag) - sum_y (zy zy^dag) zx zx^dag)
    def deriv(self, z):
        latt_shape, N = z.shape[:-1], z.shape[-1]
        deriv = np.zeros(latt_shape + (N,N), dtype=np.complex128)
        for mu in range(len(latt_shape)):
            z_fwd = np.roll(z, -1, axis=mu)
            z_bwd = np.roll(z, 1, axis=mu)
            inner_fwd = inner(z, z_fwd)
            inner_bwd = inner(z, z_bwd)
            deriv -= self.beta * (
                np.einsum('...i,...,...j->...ij', z, inner_fwd, z_fwd.conj()) +
                np.einsum('...i,...,...j->...ij', z, inner_bwd, z_bwd.conj())
            )
        deriv = deriv - deriv.swapaxes(-1,-2).conj()
        assert np.allclose(np.einsum('...ii', deriv), 0.0)
        return deriv

def _test_deriv(rng):
    N = 3
    beta = 1.0
    shape = (4,4)
    action = CPNQuarticAction(beta)
    z = make_hot_cfg(shape, N, rng=rng)
    d = 1e-6
    dz = d*(rng.normal(size=shape+(N,N)) + 1j*rng.normal(size=shape+(N,N)))
    dz = (dz - dz.swapaxes(-1,-2).conj())/2
    zp = np.einsum('...ij,...j->...i', sp.linalg.expm(dz), z)
    assert np.allclose(np.real(inner(zp,zp)), 1.0)
    old_S = action.action(z)
    deriv_S = action.deriv(z)
    new_S = action.action(zp)
    emp_dS = new_S - old_S
    thy_dS = 2*np.sum(np.einsum('...ij,...ji', deriv_S, dz))
    print(f'{emp_dS=}')
    ratio = emp_dS / thy_dS
    print(f'{ratio=}')
    assert np.isclose(ratio, 1.0, rtol=1e-4)
    print('[PASSED test_deriv]')
if __name__ == '__main__':
    _test_deriv(np.random.default_rng(238749))

### OBSERVABLES

# Geometric definition (see e.g. 1805.11058):
# q(x) = [
#   (z^dag(x+1+2) z(x+1)) (z^dag(x+1) z(x)) (z^dag(x) z(x+1+2)) +
#   (z^dag(x+2) z(x+1+2)) (z^dag(x+1+2) z(x)) (z^dag(x) z(x+2))
# ] / (2pi)
def topological_charge(z):
    assert len(z.shape) == 2+1
    z1 = np.roll(z, -1, axis=0)
    z2 = np.roll(z, -1, axis=1)
    z12 = np.roll(z1, -1, axis=1)
    q = (np.angle(inner(z12, z1) * inner(z1, z) * inner(z, z12)) +
         np.angle(inner(z2, z12) * inner(z12, z) * inner(z, z2))) / (2*np.pi)
    return q

def wflow(z, *, dt, n_step):
    action = CPNQuarticAction(1.0)
    for i in range(n_step):
        dz = dt*action.deriv(z)
        z = np.einsum('...ij,...j->...i', sp.linalg.expm(dz), z)
        S = action.action(z)
    return z

def test_topo_cfg():
    shape = (16,16)
    N = 2
    z = make_topo_cfg(shape, N)
    Q = np.sum(topological_charge(z))
    zp = wflow(z, dt=0.01, n_step=1000)
    Qp = np.sum(topological_charge(zp))
    assert np.isclose(Qp, 1.0)
    print('[PASSED topo_cfg]')
if __name__ == '__main__':
    test_topo_cfg()

def measure_2pt_susc(z):
    latt_shape = z.shape[:-1]
    assert len(latt_shape) == 2, 'specialized for 2d'
    Lx, Lt = latt_shape
    X2 = 0
    for x in range(Lx):
        for t in range(Lt):
            X2 += np.sum(np.abs(inner(z, np.roll(z, (-x,-t), axis=(0,1))))**2)
    V = Lx*Lt
    return X2 / V


### HEATBATH

def binary_search(f, val, *, yi, yf, rtol):
    assert np.all(yf > yi)
    dy = yf - yi
    val_i, val_f = f(yi), f(yf)
    while np.any(val_i > val):
        yi = np.where(val_i > val, yi-dy, yi)
        yf = np.where(val_i > val, yi, yf)
        val_i, val_f = f(yi), f(yf)
    while np.any(val_f < val):
        yi = np.where(val_f < val, yf, yi)
        yf = np.where(val_f < val, yf+dy, yf)
        val_i, val_f = f(yi), f(yf)
    y = (yf + yi) / 2
    val_p = f(y)
    while np.any(np.abs(val_p - val) > np.abs(val)*rtol):
        y = (yf + yi) / 2
        val_p = f(y)
        yf = np.where(val_p > val, y, yf)
        yi = np.where(val_p > val, yi, y)
        val_f = np.where(val_p > val, val_p, val_f)
        val_i = np.where(val_p > val, val_i, val_p)
    assert np.all(np.abs(f(y) - val) <= np.abs(val)*rtol)
    return y

def one_site_direct(Ms, *, rng):
    w, v = np.linalg.eigh(Ms)
    assert np.all(w >= 0), 'Ms must be psd'
    n_cfg, N = w.shape
    r = np.ones((n_cfg,1))
    zs = np.zeros((n_cfg, N), dtype=np.complex128)
    for i in range(N-1):
        assert np.min(np.abs(w[:,i+1:]-w[:,i:i+1])) > 0, \
            'M must be non-degenerate FORNOW'
        beta_1 = w[:,i:i+1]
        beta_j = w[:,i+1:]
        k_j = beta_1 - beta_j
        c_j = np.exp(beta_j * r**2) / k_j
        denom = beta_j[...,np.newaxis] - beta_j[...,np.newaxis,:]
        for j in range(N-i-1):
            denom[...,j,j] = 1.0
        denom = np.prod(denom, axis=-1)
        c_j /= denom
        norm = np.sum(c_j * (np.exp(k_j * r) - 1), axis=-1, keepdims=True)
        c_j /= norm
        cdf = lambda x: np.sum(c_j * (np.exp(k_j * x[...,np.newaxis]) - 1), axis=-1)
        y = rng.random(size=n_cfg)
        xi = binary_search(cdf, y, yi=np.zeros(n_cfg), yf=r[:,0], rtol=1e-6)
        print(f'{xi.shape=}')
        r -= xi[:,np.newaxis]
        thi = 2*np.pi*rng.random(size=n_cfg)
        zi = np.sqrt(xi)*np.exp(1j*thi)
        zs += zi[...,np.newaxis] * v[...,i]
    thi = 2*np.pi*rng.random(size=n_cfg)
    zi = np.sqrt(r[:,0]) * np.exp(1j*thi)
    zs += zi[...,np.newaxis] * v[...,N-1]
    return zs
