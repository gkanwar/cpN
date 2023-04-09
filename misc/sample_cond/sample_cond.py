### Sample the conditional distribution
### z ~ exp(beta z^dag (\sum_i z_i z_i^dag) z)

import matplotlib.pyplot as plt
import numpy as np
import paper_plt
paper_plt.load_latex_config()
import scipy as sp
import scipy.linalg
import scipy.interpolate
import scipy.special
import tqdm

def random_cpN(N, *, rng):
    z = rng.normal(size=N) + 1j*rng.normal(size=N)
    z /= np.sqrt(np.sum(np.abs(z)**2))
    return z

# SO(N) matrix sampled from gaussian about Id
def gaussian_soN_matrix(N, *, sigma, rng):
    A = sigma*rng.normal(size=(N,N))
    A = (A - np.transpose(A)) / 2
    return sp.linalg.expm(A)

# Metropolis update for distribution exp(z^dag M z)
def one_site_update(z, M, *, sigma, rng):
    S = - np.conj(z) @ M @ z
    N = z.shape[-1]
    X = gaussian_soN_matrix(2*N, sigma=sigma, rng=rng)
    zp = X @ np.concatenate((np.real(z), np.imag(z)))
    zp = zp[:N] + 1j*zp[N:]
    Sp = - np.conj(zp) @ M @ zp
    if rng.random() < np.exp(-Sp + S):
        return { 'cfg': zp, 'acc': True }
    else:
        return { 'cfg': z, 'acc': False }

def one_site_mcmc(beta, z_i, *, sigma, n_iter, n_therm, n_skip, rng):
    M = beta * np.einsum('xi,xj->ij', z_i, np.conj(z_i))
    N = z_i.shape[-1]
    z = random_cpN(N, rng=rng)
    acc = 0
    ens = []
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        res = one_site_update(z, M, sigma=sigma, rng=rng)
        z = res['cfg']
        acc += res['acc']
        if i >= 0 and (i+1) % n_skip == 0:
            ens.append(np.copy(z))
    return ens

def one_site_direct_N2(M, *, n_cfg, rng):
    w, v = np.linalg.eigh(M)
    assert np.all(w >= 0)
    assert len(w) == 2, 'specialized for N=2'
    assert not np.isclose(w[0], w[1]), 'eigs must be non-degenerate'
    x1 = np.linspace(0, 1, num=1000, endpoint=True)
    cdf1 = (1 - np.exp((w[0] - w[1])*x1))
    cdf1 /= cdf1[-1]
    cdf1_inv = sp.interpolate.interp1d(cdf1, x1)
    x1 = cdf1_inv(rng.random(size=n_cfg))
    assert np.all(x1 <= 1.0)
    assert np.all(x1 >= 0.0)
    x2 = 1-x1
    th1 = 2*np.pi*rng.random(size=n_cfg)
    th2 = 2*np.pi*rng.random(size=n_cfg)
    x1 = x1[:,np.newaxis]
    x2 = x2[:,np.newaxis]
    th1 = th1[:,np.newaxis]
    th2 = th2[:,np.newaxis]
    z = np.sqrt(x1)*np.exp(1j*th1)*v[:,0] + np.sqrt(x2)*np.exp(1j*th2)*v[:,1]
    return z

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

def run_N2():
    # setup
    setup_rng = np.random.default_rng(seed=1234)
    rng = np.random.default_rng()
    beta = 1.0
    sigma = 1.0
    N = 2
    z_i = np.stack([random_cpN(N, rng=setup_rng) for _ in range(4)])
    params = dict(
        n_iter=100000,
        n_skip=10,
        n_therm=100,
        rng=rng
    )

    # analytical
    M = beta * np.einsum('xi,xj->ij', z_i, np.conj(z_i))
    r = np.linspace(0, 1, num=100, endpoint=True)
    pr = np.real(
        r*np.exp(r**2 * M[0,0] + (1-r**2) * M[1,1]) * 
        sp.special.iv(0, 2*r*np.sqrt(1-r**2)*np.abs(M[0,1])))
    pr /= np.sum((pr[1:]+pr[:-1])/2)*(r[1]-r[0])
    theta = np.linspace(-np.pi, np.pi, num=100, endpoint=True)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    p_joint = (
        r_mesh*np.exp(
            r_mesh**2 * M[0,0] + (1-r_mesh**2) * M[1,1] +
            r_mesh*np.sqrt(1-r_mesh**2)*np.abs(M[0,1])
            * 2 * np.cos(theta_mesh - np.angle(M[0,1]))))
    assert np.allclose(np.imag(p_joint), 0.0)
    p_joint = np.real(p_joint)
    p_joint /= np.sum((
        p_joint[1:,1:]+p_joint[1:,:-1]+p_joint[:-1,1:]+p_joint[:-1,:-1])/4
    ) * (r[1]-r[0])*(theta[1]-theta[0])
    p_theta = np.sum((p_joint[:,1:]+p_joint[:,:-1])/2, axis=1)*(r[1]-r[0])
        

    # direct
    n_cfg = params['n_iter']//params['n_skip']
    # ensB = one_site_direct_N2(M, n_cfg=n_cfg, rng=rng)
    ensB = one_site_direct(np.tile(M, (n_cfg, 1, 1)), rng=rng)
    z1B = np.array(ensB)[:,0]
    z2B = np.array(ensB)[:,1]

    # mcmc
    ensA = one_site_mcmc(beta, z_i, sigma=sigma, **params)
    z1A = np.array(ensA)[:,0]
    z2A = np.array(ensA)[:,1]

    # plot
    fig, axes = plt.subplots(2,1, tight_layout=True)
    axes[0].hist(np.abs(z1A), bins=30, density=True, color='b')
    axes[1].hist(np.angle(z1A * np.conj(z2A)), bins=30, density=True, color='b')
    axes[0].hist(np.abs(z1B), bins=30, density=True, color='r', alpha=0.5)
    axes[1].hist(np.angle(z1B * np.conj(z2B)), bins=30, density=True, color='r', alpha=0.5)
    axes[0].plot(r, pr)
    axes[1].plot(theta, p_theta)
    fig.savefig('z_hist_N2.pdf')
    fig, axes = plt.subplots(2,1, tight_layout=True)
    axes[0].hist2d(np.abs(z1A), np.angle(z1A * np.conj(z2A)), bins=30, density=True)
    axes[1].contourf(r_mesh, theta_mesh, p_joint)
    fig.savefig('z_contour_N2.pdf')
    # fig, axes = plt.subplots(2,1, tight_layout=True)
    # axes[0].plot(np.abs(z1), marker='o')
    # axes[1].plot(np.angle(z1 * np.conj(z2)), marker='o')
    # fig.savefig('z_mcmc.pdf')

def run_N3():
    # setup
    setup_rng = np.random.default_rng(seed=1235)
    rng = np.random.default_rng()
    beta = 1.0
    sigma = 1.0
    N = 3
    z_i = np.stack([random_cpN(N, rng=setup_rng) for _ in range(4)])
    params = dict(
        n_iter=100000,
        n_skip=10,
        n_therm=100,
        rng=rng
    )
    M = beta * np.einsum('xi,xj->ij', z_i, np.conj(z_i))
    
    # direct
    n_cfg = params['n_iter']//params['n_skip']
    ensB = one_site_direct(np.tile(M, (n_cfg, 1, 1)), rng=rng)
    z1B = np.array(ensB)[:,0]
    z2B = np.array(ensB)[:,1]
    z3B = np.array(ensB)[:,2]

    # mcmc
    ensA = one_site_mcmc(beta, z_i, sigma=sigma, **params)
    z1A = np.array(ensA)[:,0]
    z2A = np.array(ensA)[:,1]
    z3A = np.array(ensA)[:,2]

    # plot
    fig, axes = plt.subplots(2,2, tight_layout=True)
    axes[0,0].hist(np.abs(z1A), bins=30, density=True, color='b')
    axes[1,0].hist(np.angle(z1A * np.conj(z2A)), bins=30, density=True, color='b')
    axes[0,1].hist(np.abs(z2A), bins=30, density=True, color='b')
    axes[1,1].hist(np.angle(z2A * np.conj(z3A)), bins=30, density=True, color='b')
    axes[0,0].hist(np.abs(z1B), bins=30, density=True, color='r', alpha=0.5)
    axes[1,0].hist(np.angle(z1B * np.conj(z2B)), bins=30, density=True, color='r', alpha=0.5)
    axes[0,1].hist(np.abs(z2B), bins=30, density=True, color='r', alpha=0.5)
    axes[1,1].hist(np.angle(z2B * np.conj(z3B)), bins=30, density=True, color='r', alpha=0.5)
    fig.savefig('z_hist_N3.pdf')
    

if __name__ == '__main__': run_N3()
