from .cpn import *

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def metropolis_update(z, *, action, sigma, rng):
    latt_shape,  N = z.shape[:-1], z.shape[-1]
    mask0 = get_checkerboard_mask(0, shape=latt_shape)
    mask1 = ~mask0
    zp = propose_update(z, sigma=sigma, rng=rng)
    u = rng.random(size=latt_shape)
    new_z = np.copy(z)
    acc = 0.0
    for mask in (mask0, mask1):
        new_z[mask] = zp[mask]
        S = action.local_action(z)
        Sp = action.local_action(new_z)
        rej = (u >= np.exp(-Sp+S)) & mask
        acc += (1.0 - np.sum(rej) / np.sum(mask)) / 2
        new_z[rej] = z[rej]
    return {
        'cfg': new_z,
        'acc': acc
    }


def run_metropolis(z0, *, action, sigma, n_iter, n_therm, n_skip, rng):
    z = z0
    S = []
    ens = []
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        res = metropolis_update(z, action=action, sigma=sigma, rng=rng)
        z = res['cfg']
        Si = action.action(z)
        if i >= 0 and (i+1) % n_skip == 0:
            S.append(Si)
            ens.append(z)
            print(f'Step {i+1}: Action {Si:.2f} Acc {100*res["acc"]:.2f}%')
    return dict(
        S=np.array(S),
        ens=np.array(ens)
    )

