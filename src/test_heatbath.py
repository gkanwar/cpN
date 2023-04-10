from lib.cpn import *
from lib.cpn_heatbath import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    beta = 3.0
    action = CPNQuarticAction(beta)
    N = 2
    shape = (32,32)
    seed = 43298
    rng = np.random.default_rng(seed)
    # z0 = make_cold_cfg(shape, N)
    # z0 = make_hot_cfg(shape, N, rng=rng)
    z0 = make_topo_cfg(shape, N)
    sigma = 0.5
    params = dict(
        action=action,
        sigma=sigma,
        n_iter=1000,
        n_therm=0,
        n_skip=10,
        rng=rng
    )
    res = run_metropolis(z0, **params)
    wflow_params = dict(dt=0.05, n_step=500)
    Q = [np.sum(topological_charge(wflow(z, **wflow_params))) for z in tqdm.tqdm(res['ens'])]
    Q = np.array(Q)

    print(res['ens'][-1])

    fig, axes = plt.subplots(2,1, tight_layout=True)
    axes[0].plot(res['S'])
    axes[0].set_ylabel('S')
    axes[1].plot(Q, marker='.')
    axes[1].set_ylabel('Q')
    fig.savefig('test_heatbath.pdf')
    
if __name__ == '__main__':
    main()
