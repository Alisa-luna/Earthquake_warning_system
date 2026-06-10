# ============================================
# train_only.py：只训练，不下载
# 数据目录：./continuous_data_3ch_v2
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from obspy import read
from scipy import signal

# ==================== 配置 ====================
DATA_DIR = "./continuous_data_3ch_v2"
MID_CHANNELS = 16
HEAD_HIDDEN = 256
LR = 0.002
T_0_WARM = 30
EPOCHS = 200
BATCH_SIZE = 512
MAX_WAVEFORMS = 500
MAX_WINDOWS_PER_WF = 500
SLIDE_STRIDE_TRAIN = 50
SLIDE_STRIDE_VAL = 100


# ==================== 模型 ====================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + x)


class TCN_Predictor_3CH(nn.Module):
    def __init__(self, input_channels=3, output_len=300):
        super().__init__()
        self.stem = CausalConv1d(input_channels, MID_CHANNELS, 7)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(MID_CHANNELS, dilation=2 ** i) for i in range(8)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(MID_CHANNELS, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, output_len),
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.tcn_blocks:
            x = blk(x)
        return self.head(self.pool(x).squeeze(-1))


# ==================== 数据集 ====================
class SlidingWindowDataset_3CH(Dataset):
    def __init__(self, data_dir, mode='train', val_split=0.1):
        self.total_len = 400

        mseed_files = sorted(glob.glob(os.path.join(data_dir, "*.mseed")))
        print(f"找到 {len(mseed_files)} 个 mseed 文件")

        self.waveforms = []
        for fp in tqdm(mseed_files, desc=f"加载{mode}数据"):
            try:
                st = read(fp)
            except:
                continue

            stations = {}
            for tr in st:
                sta = tr.stats.station
                stations.setdefault(sta, []).append(tr)

            for sta, traces in stations.items():
                comps = {}
                for tr in traces:
                    ch = tr.stats.channel
                    if ch.endswith('Z'):
                        comps['Z'] = tr
                    elif ch.endswith('N') or ch.endswith('1'):
                        comps['N'] = tr
                    elif ch.endswith('E') or ch.endswith('2'):
                        comps['E'] = tr

                if len(comps) < 3:
                    continue

                for tr in comps.values():
                    if tr.stats.sampling_rate != 100:
                        tr.resample(100)

                starts = [tr.stats.starttime for tr in comps.values()]
                ends = [tr.stats.endtime for tr in comps.values()]
                cs, ce = max(starts), min(ends)
                clen = int((ce - cs) * 100)

                if clen < self.total_len:
                    continue

                def gd(tr):
                    off = int((cs - tr.stats.starttime) * 100)
                    return signal.detrend(tr.data[off:off + clen].astype(np.float32))

                n_windows = min((clen - self.total_len) // SLIDE_STRIDE_TRAIN + 1, MAX_WINDOWS_PER_WF)

                self.waveforms.append({
                    'z': gd(comps['Z']),
                    'n': gd(comps['N']),
                    'e': gd(comps['E']),
                    'length': clen,
                    'n_windows': n_windows,
                    'station': sta
                })

                if len(self.waveforms) >= MAX_WAVEFORMS:
                    break
            if len(self.waveforms) >= MAX_WAVEFORMS:
                break

        total_windows = sum(wf['n_windows'] for wf in self.waveforms)
        print(f"加载 {len(self.waveforms)} 个台站，共 {total_windows} 个窗口")

        np.random.seed(42)
        idx = np.random.permutation(len(self.waveforms))
        sp = max(1, int(len(idx) * (1 - val_split)))

        if mode == 'train':
            self.wf_indices = idx[:sp]
        else:
            self.wf_indices = idx[sp:]

        MAX_PER_STATION = 500
        slide = SLIDE_STRIDE_TRAIN if mode == 'train' else SLIDE_STRIDE_VAL
        self.window_index = []
        for wf_idx in self.wf_indices:
            n_win = min(self.waveforms[wf_idx]['n_windows'], MAX_PER_STATION)
            for w in range(n_win):
                self.window_index.append((wf_idx, w * slide))

        print(f"{mode}: {len(self.window_index)} 个样本")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        wf_idx, start = self.window_index[idx]
        wf = self.waveforms[wf_idx]

        z_seg = wf['z'][start:start + self.total_len].copy()
        n_seg = wf['n'][start:start + self.total_len].copy()
        e_seg = wf['e'][start:start + self.total_len].copy()

        for seg in [z_seg, n_seg, e_seg]:
            seg /= (np.max(np.abs(seg)) + 1e-10)

        x = np.stack([z_seg[:100], n_seg[:100], e_seg[:100]])
        y = z_seg[100:400]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ==================== 训练 ====================
def correlation_z(y_pred, y_true):
    pred = y_pred.reshape(y_pred.shape[0], -1)
    true = y_true.reshape(y_true.shape[0], -1)
    pm = pred - pred.mean(dim=1, keepdim=True)
    tm = true - true.mean(dim=1, keepdim=True)
    num = (pm * tm).sum(dim=1)
    den = torch.sqrt((pm ** 2).sum(dim=1) * (tm ** 2).sum(dim=1))
    return (num / (den + 1e-8)).mean().item()


def train_epoch(model, loader, opt, sch, device, epoch):
    model.train()
    met = {'mse': 0, 'corr': 0}
    pbar = tqdm(loader, desc=f"Train(E{epoch + 1})")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        met['mse'] += loss.item()
        met['corr'] += correlation_z(y_pred.unsqueeze(1), y.unsqueeze(1))
        pbar.set_postfix(MSE=f"{loss.item():.3f}", Z=f"{met['corr'] / (pbar.n + 1):.3f}")
    if sch: sch.step(epoch)
    return {k: v / len(loader) for k, v in met.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    met = {'mse': 0, 'corr': 0}
    for x, y in tqdm(loader, desc="Validating"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        met['mse'] += F.mse_loss(y_pred, y).item()
        met['corr'] += correlation_z(y_pred.unsqueeze(1), y.unsqueeze(1))
    return {k: v / len(loader) for k, v in met.items()}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SlidingWindowDataset_3CH(DATA_DIR, 'train', val_split=0.1)
    val_ds = SlidingWindowDataset_3CH(DATA_DIR, 'val', val_split=0.1)

    if len(train_ds) == 0:
        print("❌ 训练集为空")
        return None
    if len(val_ds) == 0:
        print("⚠️ 验证集为空，用训练集代替")
        val_ds = train_ds

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = TCN_Predictor_3CH().to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0_WARM, T_mult=2, eta_min=1e-6)

    best_corr = -1
    for epoch in range(EPOCHS):
        train_m = train_epoch(model, train_loader, opt, sch, device, epoch)
        val_m = validate(model, val_loader, device)
        print(f"E{epoch + 1} Train Z:{train_m['corr']:.3f} Val Z:{val_m['corr']:.3f} MSE:{val_m['mse']:.4f}")
        if val_m['corr'] > best_corr + 0.01:
            best_corr = val_m['corr']
            torch.save(model.state_dict(), "tcn_3ch_best.pth")
            print(f"  ✅ 最佳 Z:{best_corr:.3f}")

    print(f"\n最佳 Z_Corr: {best_corr:.3f}")
    return model


if __name__ == "__main__":
    train()