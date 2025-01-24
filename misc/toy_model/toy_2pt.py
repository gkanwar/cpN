import analysis as al
import matplotlib.pyplot as plt
import numpy as np
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

def run_ens(meas, *, beta, Nc, n_iter, n_therm, n_meas):
    out = {k: [] for k in meas}
    # real S^{2n+2} representation
    Z = np.zeros((2, 2*(Nc+1)))
    Z[:,0] = 1.0
    acc = 0
    tot = 0
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        acc += update(Z, beta=beta)
        tot += 1
        if i >= 0 and (i+1) % n_meas == 0:
            for k,O in meas.items():
                out[k].append(O(real2cmplx(Z)))
    print(f'Final acceptance: {acc/tot}')
    return {k: np.array(v) for k,v in out.items()}

def main():
    Nc = 2
    betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] #, 7.5, 10.0, 15.0, 20.0]
    meas = {'2pt': meas_2pt}
    twopts = []
    for beta in betas:
        res = run_ens(meas, beta=beta, Nc=Nc, n_iter=25000, n_therm=100, n_meas=10)
        twopts.append(al.bootstrap(res['2pt'], Nboot=1000, f=al.rmean))
    twopts = np.transpose(twopts)
    print(f'{twopts=}')
    xs = np.linspace(0.01, 5.0)
    # ys = 1/(Nc+1) * (sp.special.gamma(Nc+2) * sp.special.gammainc(Nc+2, xs)) / (
    #     sp.special.gamma(Nc+1) * sp.special.gammainc(Nc+1, xs))
    ys = 1/Nc * (
        np.exp(-xs) * xs**(Nc+1) + (xs - Nc) * (
            sp.special.gamma(Nc+1) * sp.special.gammainc(Nc+1, xs)
        )) / (xs * sp.special.gamma(Nc) * sp.special.gammainc(Nc, xs))
    ys2 = 1 - Nc/xs + np.exp(-xs)*xs**(Nc-1) / (sp.special.gamma(Nc) * sp.special.gammainc(Nc, xs))
    fig, ax = plt.subplots(1,1)
    al.add_errorbar(twopts, ax=ax, xs=betas, label='MCMC data', color='k', marker='o', capsize=2, linestyle='')
    ax.plot(xs, ys, color='r', label='Exact result')
    ax.plot(xs, ys2, color='b', linestyle='--', label='Exact result 2')
    ax.set_ylabel(r'Two-pt fn')
    ax.set_xlabel(r'$\beta$')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
