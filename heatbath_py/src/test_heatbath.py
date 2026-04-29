from lib.cpn import *
from lib.cpn_heatbath import *

import analysis as al
import matplotlib.pyplot as plt
import numpy as np


def main():
    beta = 6.0
    action = CPNQuarticAction(beta)
    N = 3
    shape = (4,4)
    seed = 43298
    rng = np.random.default_rng(seed)
    z0 = make_cold_cfg(shape, N)
    # z0 = make_hot_cfg(shape, N, rng=rng)
    # z0 = make_topo_cfg(shape, N)
    sigma = 0.5
    params = dict(
        action=action,
        sigma=sigma,
        n_iter=100000,
        n_therm=100,
        n_skip=10,
        rng=rng
    )
    res = run_metropolis(z0, **params)
    # wflow_params = dict(dt=0.05, n_step=500)
    # Q = [np.sum(topological_charge(wflow(z, **wflow_params))) for z in tqdm.tqdm(res['ens'])]
    # Q = np.array(Q)
    # X2 = [measure_2pt_susc(z) for z in tqdm.tqdm(res['ens'])]

    fig, ax = plt.subplots(1,1, tight_layout=True, figsize=(6,4))
    ax.plot(res['S'])
    ax.set_ylabel('S')
    # axes[1].plot(Q, marker='.')
    # axes[1].set_ylabel('Q')
    # axes[2].plot(X2)
    # axes[2].set_ylabel('susc')
    fig.savefig('test_heatbath.pdf')

    est_S = al.bootstrap(res['S'], Nboot=100, f=al.rmean)
    print(f'{est_S=}')

    np.save(f'cpn_b{beta:.1f}.npy', res['ens'])
    
if __name__ == '__main__':
    main()
