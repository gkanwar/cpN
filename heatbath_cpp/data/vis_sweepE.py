import analysis as al
import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm

def main():
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    beta_strs = ['0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5', '6']
    us = []
    for beta, beta_s in zip(betas, beta_strs):
        u = np.loadtxt(f'cpn_b{beta_s}_L64_Nc3_sweepE_u.dat')
        us.append(al.bootstrap(al.bin_data(u, binsize=100)[1], Nboot=1000, f=al.rmean))
    us = np.stack(us, axis=-1)
    print(f'{us=}')
    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    style = dict(marker='o', capsize=4, linestyle='', fillstyle='none')
    al.add_errorbar(us, xs=betas, ax=ax, **style)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\langle E \rangle$')
    ax.set_ylim(0, 2.0)
    ax.set_xlim(0, 6.5)
    fig.set_tight_layout(True)
    fig.savefig('cpn_L64_sweepE.pdf')
    plt.show()

if __name__ == '__main__':
    main()
