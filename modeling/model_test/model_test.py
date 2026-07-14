import pytest
import numpy as np
import torch

# Import directly from parent/sibling directories thanks to the standard layout
from config import ModelConfig  # <-- the actual config class name
from base.tcn_q import TCNQuantile, pinball_loss_torch

# =====================================================================
# FIXTURES
# =====================================================================
@pytest.fixture
def real_cfg():
    """Build the real config from config.py."""
    cfg = ModelConfig()

    # shrink the params for the test run so it stays fast
    cfg.tcn_epochs = 2
    cfg.tcn_batch = 4
    cfg.seed = 42
    return cfg

@pytest.fixture
def dummy_data():
    """Create synthetic data with the same structure as the real solar data."""
    np.random.seed(42)
    n_samples = 8
    seq_len = 12       # time-series length (window size)
    n_features = 39    # number of features
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32) * 20.0
    return X, y

# =====================================================================
# UNIT TESTS
# =====================================================================

def test_pinball_loss():
    """Check the loss function returns a non-negative scalar."""
    pred = torch.tensor([[5.0, 10.0, 15.0]], dtype=torch.float32)
    target = torch.tensor([10.0], dtype=torch.float32)
    quantiles = [0.1, 0.5, 0.9]
    
    loss = pinball_loss_torch(pred, target, quantiles)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

def test_tcn_pipeline(real_cfg, dummy_data):
    """Test the whole flow: init -> fit -> predict."""
    X, y = dummy_data
    n_features = X.shape[2]

    # 1. init
    model = TCNQuantile(real_cfg, n_features=n_features)
    assert model.model_ is None

    # 2. train for 2 epochs
    model.fit(X, y)
    assert model.model_ is not None

    # 3. predict
    preds = model.predict(X)
    assert isinstance(preds, dict)

    # check every configured quantile is returned
    assert list(preds.keys()) == list(real_cfg.quantiles)

    # check monotonicity: q10 <= q50 <= q90
    # (ensures the np.sort step works correctly)
    if len(real_cfg.quantiles) >= 3:
        q_low = real_cfg.quantiles[0]
        q_med = real_cfg.quantiles[1]
        q_high = real_cfg.quantiles[2]
        assert np.all(preds[q_low] <= preds[q_med])
        assert np.all(preds[q_med] <= preds[q_high])

def test_empty_input_fallback(real_cfg, dummy_data):
    """Edge case: if the test array is empty (0 samples), does the code crash?"""
    X, y = dummy_data
    n_features = X.shape[2]

    model = TCNQuantile(real_cfg, n_features=n_features).fit(X, y)

    # empty array of shape (0, seq_len, n_features)
    empty_X = np.empty((0, X.shape[1], n_features), dtype=np.float32)
    preds = model.predict(empty_X)

    # must return an empty array of shape (0,) instead of crashing
    for q in real_cfg.quantiles:
        assert preds[q].shape == (0,)


