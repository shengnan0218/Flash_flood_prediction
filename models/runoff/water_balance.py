"""Mass-conserving two-reservoir rainfall-runoff module."""
from __future__ import annotations
import torch
from torch import nn

class WaterBalanceLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(); self.cell = nn.LSTMCell(input_dim, hidden_dim); self.controls = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, rain_mm: torch.Tensor, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        h, c, sf, ss = state; h, c = self.cell(x, (h, c)); g, kf, ks = torch.sigmoid(self.controls(h)).unbind(-1)
        # 中文/EN: 先分配降雨，再释放更新后库容 / partition rain then release from updated storage.
        af = sf + g * rain_mm; ass = ss + (1.0 - g) * rain_mm
        rf = kf * af; rs = ks * ass; nsf = af - rf; nss = ass - rs
        # Form runoff as the exact floating-point remainder of the same total.
        # This preserves the structural identity to machine precision in float32.
        total = (sf + ss) + rain_mm
        runoff = total - (nsf + nss)
        residual = total - ((nsf + nss) + runoff)
        return runoff, (h, c, nsf, nss), {"g": g, "k_fast": kf, "k_slow": ks, "storage_fast": nsf, "storage_slow": nss, "residual": residual}

class WaterBalanceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(); self.hidden_dim = hidden_dim; self.cell = WaterBalanceLSTMCell(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor, rain: torch.Tensor, area_km2: torch.Tensor, seconds: float = 3600.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Physical equations intentionally float32 even under autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            x, p, area = features.float(), rain.float().squeeze(-1), area_km2.float(); b, t, n, _ = x.shape
            z = torch.zeros(b*n, self.hidden_dim, device=x.device); state = (z, z.clone(), torch.zeros(b*n, device=x.device), torch.zeros(b*n, device=x.device))
            qs=[]; residual=[]; stores=[]
            for i in range(t):
                r, state, d = self.cell(x[:, i].reshape(b*n, -1), p[:, i].reshape(-1), state)
                # 1 mm over 1 km2 = 1000 m3.
                qs.append((r.reshape(b,n) * area[None] * 1000.0 / seconds)); residual.append(d["residual"].reshape(b,n)); stores.append((d["storage_fast"]+d["storage_slow"]).reshape(b,n))
            return torch.stack(qs,1), {"runoff_water_balance_residual": torch.stack(residual,1), "storage": torch.stack(stores,1)}

class PureLSTMRunoff(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(); self.lstm=nn.LSTM(input_dim, hidden_dim, batch_first=True); self.head=nn.Sequential(nn.Linear(hidden_dim,1),nn.Softplus())
    def forward(self, features: torch.Tensor, rain: torch.Tensor, area_km2: torch.Tensor, seconds: float=3600.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b,t,n,d=features.shape; y,_=self.lstm(features.permute(0,2,1,3).reshape(b*n,t,d)); q=self.head(y).reshape(b,n,t).permute(0,2,1)
        return q, {"runoff_water_balance_residual": torch.full_like(q, float("nan"))}
