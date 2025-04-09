import numpy as np

from cymmd.mmd import CythonMMD


def main():
    sample_dim = 4

    # Generate random samples
    x = np.random.randn(10_000, sample_dim)
    y = (
        np.random.randn(10_000, sample_dim) + 1
    )  # The second sample group has a mean offset of 1

    # Calculate MMD
    res = CythonMMD.calculate(x, y, block_size=500)
    print(f"MMD between x and y: {res:.4f}")


if __name__ == "__main__":
    main()
