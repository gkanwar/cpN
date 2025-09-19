import analysis as al
import matplotlib.pyplot as plt
import numpy as np

def Phi_ij(ens):
    """Construct timeslice averages of Phi_ij operator"""
    return np.einsum('nxti,nxtj->ntij', ens, np.conj(ens)) / ens.shape[1]

def action(ens):
    n_cfg, Lx, Lt, Nc = ens.shape
    S = 0
    for mu in range(2):
        zfwd = np.conj(np.roll(ens, -1, axis=1+mu))
        S = S + np.mean(1 - np.abs(np.sum(ens * zfwd, axis=-1))**2, axis=(1,2))
    return S

def main():
    L = 64
    beta = 4.5
    Nc = 3
    u = np.loadtxt(f'cpn_b{beta:.1f}_L{L}_Nc{Nc}_u.dat')[9::10]
    n_cfg = len(u)
    ens = np.fromfile(f'cpn_b{beta:.1f}_L{L}_Nc{Nc}_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)
    assert np.allclose(np.sum(np.abs(ens)**2, axis=-1), 1.0), 'cpn vectors must be normalized'
    Lx, Lt = ens.shape[1:3]

    # check action values:
    u2 = action(ens)
    print(u[:10])
    print(u2[:10])
    # assert np.allclose(u, u2)

    fig, ax = plt.subplots(1,1, figsize=(8,3))
    ax.plot(u)
    ax.plot(u2)
    plt.show()

    Oij = Phi_ij(ens)
    Gij = np.stack([np.mean(np.roll(Oij, t, axis=1) * Oij.swapaxes(-1,-2), axis=1) for t in range(Lt)], axis=1)
    Oij_vac = np.mean(Oij, axis=1)
    print(f'{Gij.shape=} {Oij_vac.shape=}')
    # unsubtracted 2pt function
    # Gij_est1 = np.stack(al.bootstrap(Gij, Nboot=100, f=al.rmean))
    # vacuum-subtracted 2pt function
    conn_2pt = lambda G, Ovac: al.rmean(G) - al.rmean(Ovac)**2
    # effective mass estimating the exponential decay constant of conn_2pt
    conn_2pt_meff = lambda G, Ovac: np.log(conn_2pt(G, Ovac)[:-1]) - np.log(conn_2pt(G, Ovac)[1:])
    Gij_est2 = np.stack(al.bootstrap(Gij, Oij_vac, Nboot=100, f=conn_2pt))
    mij_est2 = np.stack(al.bootstrap(Gij, Oij_vac, Nboot=100, f=conn_2pt_meff))

    fig, axes = plt.subplots(2,Nc, figsize=(8,3))
    xs = np.arange(Lt)
    for i,ax in enumerate(axes[0]):
        ax.set_yscale('log')
        al.add_errorbar(Gij_est2[...,0,i], ax=ax, xs=xs)
    for i,ax in enumerate(axes[1]):
        ax.set_ylim(0.1, 0.3)
        ax.set_xlabel('$t$')
        al.add_errorbar(mij_est2[...,0,i], ax=ax, xs=xs[1:])
    axes[1,0].set_ylabel(r'$m_{\mathrm{eff}}$')
    plt.show()

if __name__ == '__main__':
    main()
