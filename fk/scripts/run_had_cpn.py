import argparse
import matplotlib.pyplot as plt
import torch
import tqdm.auto as tqdm

from fk.hmc import hmc_traj
from fk.theory import CPNQuarticTheory
from fk.util import project_cpn_u1, sample_hot

if torch.cuda.is_available():
    torch.set_default_device('cuda')
torch.set_default_dtype(torch.double)


def _flat_var(x):
    return x.flatten(1).var(dim=1, unbiased=False).mean()


def _flat_norm(x):
    return x.flatten(1).norm(dim=1).mean()


def _print_stats(step, x, b):
    b_proj = project_cpn_u1(b, x, dim=1)
    b_norm = _flat_norm(b)
    b_var = _flat_var(b)
    b_proj_norm = _flat_norm(b_proj)
    b_proj_var = _flat_var(b_proj)
    print(
        f'{step:4d} '
        f'b_norm={b_norm.item():.8e} '
        f'b_var={b_var.item():.8e} '
        f'b_proj_norm={b_proj_norm.item():.8e} '
        f'b_proj_var={b_proj_var.item():.8e}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--Nc', type=int, required=True)
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--n_leap', type=int, required=True)
    parser.add_argument('--dtau', type=float, required=True)
    parser.add_argument('--n_therm', type=int, default=0)
    parser.add_argument('--n_traj', type=int, required=True)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    torch.set_default_dtype(torch.double)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    base_theory = CPNQuarticTheory(args.beta)
    x = sample_hot(1, (args.L, args.L), n=2*args.Nc)
    b = torch.zeros_like(x)

    def had_traj(x, delta):
        class ShiftedTheory:
            def energy(self, x):
                return base_theory.energy(x)

            def action(self, x):
                return (args.beta + delta) * self.energy(x)

            def grad_action(self, x):
                return (args.beta + delta) * base_theory.gradE(x)

        res = hmc_traj(x, theory=ShiftedTheory(), n_leap=args.n_leap, dtau=args.dtau, ar=False)
        return res['x']

    delta = torch.zeros(())
    delta_tangent = torch.ones(())

    # thermalize
    for i in tqdm.tqdm(range(args.n_therm)):
        res = hmc_traj(x, theory=base_theory, n_leap=args.n_leap, dtau=args.dtau, ar=True)
        x = res['x']
        # E = base_theory.energy(x).mean().item()
        # print(f'{i:4d} {E=:.4f}')
        

    print(
        f'# beta={args.beta} Nc={args.Nc} L={args.L} '
        f'n_leap={args.n_leap} dtau={args.dtau} '
        f'n_therm={args.n_therm} n_traj={args.n_traj}'
    )
    print('# step b_norm b_var b_proj_norm b_proj_var')
    _print_stats(0, x, b)
    b_norm = []
    b_proj_norm = []
    for step in range(1, args.n_traj + 1):
        x, b = torch.func.jvp(had_traj, (x, delta), (b, delta_tangent))
        b_proj = project_cpn_u1(b, x, dim=1)
        b_norm.append(_flat_norm(b))
        b_proj_norm.append(_flat_norm(b_proj))
        _print_stats(step, x, b)

    fig, ax = plt.subplots(1,1)
    ax.plot(b_norm, label='b norm')
    ax.plot(b_proj_norm, label='b proj norm')
    ax.legend()
    ax.set_yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
