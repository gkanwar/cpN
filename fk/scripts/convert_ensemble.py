import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fname', type=str, required=True)
    parser.add_argument('--out_fname', type=str, required=True)
    parser.add_argument('--Nc', type=int, required=True)
    parser.add_argument('--L', type=int, required=True)
    args = parser.parse_args()

    Nc, L = args.Nc, args.L
    z = np.fromfile(args.in_fname, dtype=np.complex128).reshape(-1, L, L, Nc)
    x = np.transpose(z, (0, 3, 1, 2))
    x = np.concatenate([np.real(x), np.imag(x)], axis=1)
    np.save(args.out_fname, x)

if __name__ == '__main__':
    main()
