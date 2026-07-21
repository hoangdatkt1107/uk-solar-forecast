from __future__ import annotations
import numpy as np

def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends,"mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _build_torch(cfg, n_features: int, n_quantiles: int):
    import torch
    import torch.nn as nn

    class Chomp1d(nn.Module):
        def __init__(self, pad):
            super().__init__()
            self.pad = pad
        def forward(self, input):
            return input[:,:,: - self.pad].contiguous() if self.pad else input
        
    class TemporalBlock(nn.Module):
        def __init__(self, ci, co, k, d, dropout):
            super().__init__()
            pad = (k-1) * d
            self.net = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ci, co, kernel_size=k, padding=pad, dilation=d)),
                Chomp1d(pad=pad), nn.ReLU(), nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Conv1d(co, co, kernel_size=k, padding=pad, dilation=d)),
                Chomp1d(pad=pad), nn.ReLU(), nn.Dropout(dropout)
            )
            self.down = nn.Conv1d(ci, co, kernel_size=1) if ci != co else None
            self.relu = nn.ReLU() 

        def forward(self, x):
            out = self.net(x)
            res = x if self.down is None else self.down(x)
            return self.relu(out + res)
    
    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ci = [], n_features
            for i, co in enumerate(cfg.tcn_channels):
                layers.append(TemporalBlock(ci=ci, co=co, k=cfg.tcn_kernel, d=2**i, dropout=cfg.tcn_dropout))
                ci = co
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(ci, n_quantiles)

        def forward(self, x):
            h = self.tcn(x.transpose(1,2))
            return self.head(h[:, :, -1])
        
    return TCN()

def pinball_loss_torch(pred, target, quantiles):
    import torch
    target = target.unsqueeze(1)
    q = torch.tensor(quantiles, device=pred.device).view(1,-1)
    residual = target - pred
    loss = torch.mean(torch.maximum(q * residual, (q-1) * residual))
    return loss

class TCNQuantile:
    def __init__(self, cfg, n_features:int):
        self.cfg = cfg
        self.quantiles = tuple(cfg.quantiles)
        self.n_features = n_features
        self.model_ = None
    
    def build(self):
        """construct the untrained structure for reloading saved weights"""
        import torch
        torch.set_num_threads(self.cfg.torch_threads)
        self.model_ = _build_torch(self.cfg, self.n_features, len(self.quantiles))
        return self
    
    def fit(self, seqs: np.ndarray, y: np.ndarray,
            val: tuple | None = None) -> TCNQuantile:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from loguru import logger
        torch.set_num_threads(self.cfg.torch_threads)
        torch.manual_seed(self.cfg.seed)
        dev = device()
        self.model_ = _build_torch(self.cfg, self.n_features, len(self.quantiles)).to(dev)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.cfg.tcn_lr)
        ds = TensorDataset(torch.from_numpy(seqs), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.cfg.tcn_batch, shuffle=True, drop_last=False)
        epochs = self.cfg.tcn_epochs
        for epoch in range(epochs):
            self.model_.train()
            total_loss, num_of_batch = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = pinball_loss_torch(self.model_(xb), yb, self.quantiles)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                num_of_batch += 1
            if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs -1:
                logger.info(f"TCN{dev} epoch {epoch + 1}/{epochs} -- Loss={total_loss / max(num_of_batch,1)}") 

        return self
    
    def predict(self, seqs: np.ndarray) -> dict[float,np.ndarray]:
        import torch
        torch.set_num_threads(self.cfg.torch_threads)
        dev = next(self.model_.parameters()).device
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(seqs), 1024):
                xb = torch.from_numpy(seqs[i: i+1024]).to(dev)
                outs.append(self.model_(xb).cpu().numpy())
        pred = np.concatenate(outs, axis=0) if outs else np.empty((0,len(self.cfg.quantiles)))
        pred = np.sort(pred, axis=1)
        return {q: pred[:, i].astype("float32") for i, q in enumerate(self.quantiles)}


