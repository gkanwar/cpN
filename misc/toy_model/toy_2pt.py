import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import tqdm.auto as tqdm

def action(z, *, beta):
    assert z.shape[0] == 2
    return -beta * np.abs(np.sum(z[0] * np.conj(z[1]), axis=-1))**2

def update(Z, *, beta):
    N = Z.shape[-1]
    S = action(Z, beta=beta)
    acc = 0
    tot = 0
    for x in range(Z.shape[0]):
        Zx = Z[x]
        for i in range(N):
            for j in range(i+1, N):
                Zxi, Zxj = Zx[i], Zx[j]
                th = np.random.normal()
                Zpxi = np.cos(th)*Zxi + np.sin(th)*Zxj
                Zpxj = np.cos(th)*Zxj - np.sin(th)*Zxi
                Zx[i], Zx[j] = Zpxi, Zpxj
                Sp = action(real2cmplx(Z), beta=beta)
                tot += 1
                if np.random.random() < np.exp(-Sp+S):
                    acc += 1
                    S = Sp
                else:
                    Zx[i], Zx[j] = Zxi, Zxj
    return acc/tot

def real2cmplx(Z):
    return Z[:,::2] + 1j * Z[:,1::2]

def meas_2pt(z):
    assert z.shape[0] == 2
    return np.abs(np.sum(z[0] * np.conj(z[1]), axis=-1))**2

def meas_2pt_ij(z):
    assert z.shape[0] == 2
    return np.einsum('i,j,i,j->ij', z[0], np.conj(z[0]), np.conj(z[1]), z[1])

def meas_1pt_var(z):
    assert z.shape[0] == 2
    return np.einsum('i,j->ij', np.abs(z[0])**2, np.abs(z[0])**2)

def run_ens(meas, *, beta, Nc, n_iter, n_therm, n_meas):
    out = {k: [] for k in meas}
    # real S^{2n+2} representation
    Z = np.zeros((2, 2*(Nc+1)))
    Z[:,0] = 1.0
    acc = 0
    tot = 0
    ens = []
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        acc += update(Z, beta=beta)
        tot += 1
        if i >= 0 and (i+1) % n_meas == 0:
            for k,O in meas.items():
                out[k].append(O(real2cmplx(Z)))
            ens.append(np.copy(real2cmplx(Z)))
    print(f'Final acceptance: {acc/tot}')
    res = {k: np.array(v) for k,v in out.items()}
    res['ens'] = ens
    return res

def main():
    Nc = 2
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    meas = {'2pt': meas_2pt, '2ptij': meas_2pt_ij, '1ptvar': meas_1pt_var}
    twopts = []
    twopts_diag = []
    twopts_off = []
    onept_var_diag = []
    onept_var_off = []
    for beta in betas:
        _cache = f'cache/res_b{beta:0.2f}.npy'
        if not os.path.exists(_cache):
            res = run_ens(meas, beta=beta, Nc=Nc, n_iter=25000, n_therm=5000, n_meas=10)
            np.save(_cache, res)
        else:
            res = np.load(_cache, allow_pickle=True).item()
        twopts.append(al.bootstrap(res['2pt'], Nboot=1000, f=al.rmean))
        twopts_diag_meas = res['2ptij'].trace(axis1=-1,axis2=-2)/(Nc+1)
        twopts_diag.append(al.bootstrap(twopts_diag_meas, Nboot=1000, f=al.rmean))
        twopts_off_meas = (np.sum(res['2ptij'], axis=(-1,-2)) - (Nc+1)*twopts_diag_meas)/(Nc*(Nc+1))
        twopts_off.append(al.bootstrap(twopts_off_meas, Nboot=1000, f=al.rmean))
        onept_var_diag_meas = res['1ptvar'].trace(axis1=-1,axis2=-2)/(Nc+1)
        onept_var_off_meas = (np.sum(res['1ptvar'], axis=(-1,-2)) - (Nc+1)*onept_var_diag_meas)/(Nc*(Nc+1))
        onept_var_diag_meas = al.bin_data(onept_var_diag_meas, binsize=10)[1]
        onept_var_off_meas = al.bin_data(onept_var_off_meas, binsize=10)[1]
        onept_var_diag.append(al.bootstrap(onept_var_diag_meas, Nboot=1000, f=al.rmean))
        onept_var_off.append(al.bootstrap(onept_var_off_meas, Nboot=1000, f=al.rmean))
    twopts = np.transpose(twopts)
    twopts_diag = np.transpose(twopts_diag)
    twopts_off = np.transpose(twopts_off)
    onept_var_diag = np.transpose(onept_var_diag)
    onept_var_off = np.transpose(onept_var_off)

    # figs
    style_mcmc = dict(label='MCMC data', color='k', marker='o', capsize=2, linestyle='')
    style_exact = dict(color='r', label='Exact result')
    xs = np.linspace(0.01, np.max(betas)) # for exact traces

    # twopts
    fig, axes = plt.subplots(3,1, figsize=(6,6), sharex=True)
    ax = axes[0]
    al.add_errorbar(twopts, ax=ax, xs=betas, **style_mcmc)
    exact_ys = 1 - Nc/xs + np.exp(-xs)*xs**(Nc-1) / (sp.special.gamma(Nc) * sp.special.gammainc(Nc, xs))
    ax.plot(xs, exact_ys, **style_exact)
    ax.set_ylabel(r'Two-pt (traced)')
    ax.legend()
    ax = axes[1]
    al.add_errorbar(twopts_diag, ax=ax, xs=betas, **style_mcmc)
    ax.set_ylabel(r'Two-pt (diag)')
    ax = axes[2]
    al.add_errorbar(twopts_off, ax=ax, xs=betas, **style_mcmc)
    ax.set_ylabel(r'Two-pt (off-diag)')
    ax.set_xlabel(r'$\beta$')
    fig.set_tight_layout(True)

    # onept var
    fig, axes = plt.subplots(2,1, figsize=(6,4), sharex=True)
    ax = axes[0]
    al.add_errorbar(onept_var_diag, ax=ax, xs=betas, **style_mcmc)
    exact_ys = 2*np.ones_like(xs)/((Nc+1)*(Nc+2))
    ax.plot(xs, exact_ys, **style_exact)
    ax.set_ylim(exact_ys[0]*0.90, exact_ys[1]*1.10)
    ax.set_ylabel(r'One-pt var (diag)')
    ax = axes[1]
    al.add_errorbar(onept_var_off, ax=ax, xs=betas, **style_mcmc)
    exact_ys = np.ones_like(xs)/((Nc+1)*(Nc+2))
    ax.plot(xs, exact_ys, **style_exact)
    ax.set_ylim(exact_ys[0]*0.90, exact_ys[1]*1.10)
    ax.set_ylabel(r'One-pt var (off)')
    ax.set_xlabel(r'$\beta$')
    fig.set_tight_layout(True)

    plt.show()

if __name__ == '__main__':
    main()
