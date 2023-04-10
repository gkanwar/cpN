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
    for mask in (mask0, mask1):
        new_z[mask] = zp[mask]
        S = action.local_action(z)
        Sp = action.local_action(new_z)
        acc = u[mask] < np.exp(-Sp[mask]+S[mask])
        ind_rej = tuple(np.transpose(np.argwhere(mask)[~acc]))
        new_z[ind_rej] = z[ind_rej]
    return new_z


def run_metropolis(z0, *, action, sigma, n_iter, n_therm, n_skip, rng):
    z = z0
    S = []
    ens = []
    for i in tqdm.tqdm(range(-n_therm, n_iter)):
        zp = metropolis_update(z, action=action, sigma=sigma, rng=rng)
        Si = action.action(zp)
        if i >= 0 and (i+1) % n_skip == 0:
            S.append(Si)
            ens.append(zp)
    return dict(
        S=np.array(S),
        ens=np.array(ens)
    )

