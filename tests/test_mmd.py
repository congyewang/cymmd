import pytest
import numpy as np
import torch
from cymmd.mmd import CythonMMD


class TestCythonMMD:
    def test_median_trick_numpy(self):
        # Test median trick with numpy array
        x = np.random.randn(100, 10)
        sigma = CythonMMD.median_trick(x)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_median_trick_torch(self):
        # Test median trick with torch tensor
        x = torch.randn(100, 10)
        sigma = CythonMMD.median_trick(x)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_calculate_numpy(self):
        # Test MMD calculation with numpy arrays
        x = np.random.randn(100, 5)
        y = np.random.randn(150, 5)
        mmd = CythonMMD.calculate(x, y)
        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_calculate_torch(self):
        # Test MMD calculation with torch tensors
        x = torch.randn(100, 5)
        y = torch.randn(150, 5)
        mmd = CythonMMD.calculate(x, y)
        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_calculate_1d_inputs(self):
        # Test with 1D inputs
        x = np.random.randn(100)
        y = np.random.randn(150)
        mmd = CythonMMD.calculate(x, y)
        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_calculate_with_sigma(self):
        # Test with custom sigma
        x = np.random.randn(100, 5)
        y = np.random.randn(150, 5)
        mmd = CythonMMD.calculate(x, y, sigma=1.0, use_median_trick=False)
        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_calculate_block_size(self):
        # Test with different block size
        x = np.random.randn(1000, 5)
        y = np.random.randn(1000, 5)
        mmd = CythonMMD.calculate(x, y, block_size=500)
        assert isinstance(mmd, float)
        assert mmd >= 0

    def test_calculate_same_distribution(self):
        # MMD should be close to 0 for same distribution
        rng = np.random.RandomState(42)
        x = rng.randn(10_000, 5)
        y = rng.randn(10_000, 5)  # Different samples, same distribution
        mmd = CythonMMD.calculate(x, y)
        assert mmd < 0.05  # Small value expected for same distribution

    def test_calculate_different_distribution(self):
        # MMD should be larger for different distributions
        x = np.random.randn(1000, 5)  # Standard normal
        y = np.random.randn(1000, 5) * 2 + 3  # Different mean and variance
        mmd = CythonMMD.calculate(x, y)
        assert mmd > 0.1  # Larger value expected for different distributions

    def test_empty_arrays(self):
        # Test with empty arrays - should raise exception
        x = np.array([]).reshape(0, 5)
        y = np.random.randn(10, 5)
        with pytest.raises(Exception):
            CythonMMD.calculate(x, y)

    def test_single_sample(self):
        # Test with single sample in each array
        x = np.random.randn(1, 5)
        y = np.random.randn(1, 5)
        mmd = CythonMMD.calculate(x, y)
        assert isinstance(mmd, float)
