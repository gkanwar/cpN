import analysis as al
import matplotlib.pyplot as plt
import numpy as np

def main():
    L = 8
    beta = 4.5
    Nc = 3
    u = np.loadtxt(f'tmp_u.dat')[9::10]
    n_cfg = len(u)
    ens = np.fromfile(f'tmp_ens.dat', dtype=np.complex128).reshape(n_cfg, L, L, Nc)

    fig, ax = plt.subplots(1,1, figsize=(8,3))
    ax.plot(u)

    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(np.linalg.norm(ens[0], axis=-1))
    axes[1].imshow(np.angle(ens[0,:,:,0] / ens[0,:,:,1]))
    plt.show()

if __name__ == '__main__':
    main()
