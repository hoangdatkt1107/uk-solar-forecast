import pytest
import numpy as np
import torch

# Import trực tiếp từ các thư mục cha/anh em nhờ cấu trúc quy chuẩn
from config import ModelConfig  # <-- Tên class config thực tế của bạn
from base.tcn_q import TCNQuantile, pinball_loss_torch

# =====================================================================
# FIXTURES
# =====================================================================
@pytest.fixture
def real_cfg():
    """Khởi tạo cấu hình thực tế từ file config.py của bạn."""
    cfg = ModelConfig()
    
    # Ép các tham số nhỏ lại khi chạy test để không bị chậm máy
    cfg.tcn_epochs = 2  
    cfg.tcn_batch = 4
    cfg.seed = 42
    return cfg

@pytest.fixture
def dummy_data():
    """Tạo dữ liệu giả lập có cấu trúc giống hệt dữ liệu solar thực tế."""
    np.random.seed(42)
    n_samples = 8
    seq_len = 12       # Độ dài chuỗi thời gian (window size)
    n_features = 39    # Số lượng feature của bạn
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32) * 20.0
    return X, y

# =====================================================================
# UNIT TESTS
# =====================================================================

def test_pinball_loss():
    """Kiểm tra xem hàm tính loss có trả về scalar và không âm không."""
    pred = torch.tensor([[5.0, 10.0, 15.0]], dtype=torch.float32)
    target = torch.tensor([10.0], dtype=torch.float32)
    quantiles = [0.1, 0.5, 0.9]
    
    loss = pinball_loss_torch(pred, target, quantiles)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

def test_tcn_pipeline(real_cfg, dummy_data):
    """Test toàn bộ quy trình: Khởi tạo -> Fit -> Predict."""
    X, y = dummy_data
    n_features = X.shape[2]
    
    # 1. Khởi tạo
    model = TCNQuantile(real_cfg, n_features=n_features)
    assert model.model_ is None
    
    # 2. Huấn luyện thử 2 epochs
    model.fit(X, y)
    assert model.model_ is not None
    
    # 3. Dự đoán
    preds = model.predict(X)
    assert isinstance(preds, dict)
    
    # Kiểm tra xem có đủ các đầu ra tương ứng với cấu hình không
    assert list(preds.keys()) == list(real_cfg.quantiles)
    
    # Kiểm tra tính đơn điệu (Monotone): q10 <= q50 <= q90
    # Đảm bảo hàm np.sort của bạn đang hoạt động chuẩn xác
    if len(real_cfg.quantiles) >= 3:
        q_low = real_cfg.quantiles[0]
        q_med = real_cfg.quantiles[1]
        q_high = real_cfg.quantiles[2]
        assert np.all(preds[q_low] <= preds[q_med])
        assert np.all(preds[q_med] <= preds[q_high])

def test_empty_input_fallback(real_cfg, dummy_data):
    """Test thế chân tường: Nếu mảng test rỗng (0 samples), code có sập không."""
    X, y = dummy_data
    n_features = X.shape[2]
    
    model = TCNQuantile(real_cfg, n_features=n_features).fit(X, y)
    
    # Mảng rỗng kích thước (0, seq_len, n_features)
    empty_X = np.empty((0, X.shape[1], n_features), dtype=np.float32)
    preds = model.predict(empty_X)
    
    # Phải trả về mảng rỗng shape (0,) thay vì crash
    for q in real_cfg.quantiles:
        assert preds[q].shape == (0,)


