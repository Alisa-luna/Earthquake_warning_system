# ============================================
# 最终版：直接预测波形 + PCA 方案
# Optuna最优参数，目标 Z_Corr: 0.54+
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
from sklearn.decomposition import PCA
import pickle

# ==================== Optuna最优参数 ====================
N_DILATED = 7
MID_CHANNELS = 16
HEAD_LAYERS = 3
HEAD_HIDDEN = 768
LR = 0.00256
NE_WEIGHT = 0.2
T_0_WARM = 50
EPOCHS = 300
BATCH_SIZE = 512
N_PCA_COEFFS = 20


# ==================== 编码器（直接输出300点三分量波形） ====================
class Encoder_Waveform(nn.Module):
    def __init__(self, input_channels=3, output_len=300):
        super().__init__()

        # 特征提取
        self.stem = nn.Sequential(nn.Conv1d(input_channels, MID_CHANNELS, 7, padding=3), nn.ReLU())
        dilations = [2 ** i for i in range(N_DILATED)]
        self.dilated_blocks = nn.ModuleList([
            self._make_block(MID_CHANNELS, d) for d in dilations
        ])
        self.time_segment = nn.AvgPool1d(25, 25)
        self.multi_freq = nn.ModuleList([
            nn.Conv1d(MID_CHANNELS, max(4, MID_CHANNELS // 2), 3, padding=1),
            nn.Conv1d(MID_CHANNELS, max(4, MID_CHANNELS // 2), 3, padding=2, dilation=2),
            nn.Conv1d(MID_CHANNELS, max(4, MID_CHANNELS // 2), 5, padding=4, dilation=2),
        ])

        common_dim = max(4, MID_CHANNELS // 2) * 3 * 4

        # 波形生成头
        layers = []
        in_dim = common_dim
        for _ in range(HEAD_LAYERS - 1):
            layers.extend([nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU()])
            in_dim = HEAD_HIDDEN
        layers.append(nn.Linear(in_dim, 3 * output_len))
        self.waveform_gen = nn.Sequential(*layers)

    def _make_block(self, ch, d):
        return nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(),
            nn.Conv1d(ch, ch, 1), nn.ReLU(),
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.dilated_blocks:
            x = blk(x) + x
        x = self.time_segment(x)
        fs = []
        for conv in self.multi_freq:
            f = conv(x)
            f = f[:, :, :4] if f.shape[-1] >= 4 else F.pad(f, (0, 4 - f.shape[-1]))
            fs.append(f)
        flat = torch.cat(fs, dim=1).flatten(1)
        return self.waveform_gen(flat).view(-1, 3, 300)


# ==================== 数据集 ====================
class MseedRandomDataset(Dataset):
    def __init__(self, data_dir, mode='train', val_split=0.1, max_waveforms=500, target_sr=100.0):
        self.mode = mode
        self.target_sr = target_sr
        self.input_len = 100
        self.output_len = 300
        self.total_len = 400

        mseed_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.mseed"), recursive=True))
        print(f"找到 {len(mseed_files)} 个 mseed 文件")

        self.waveforms = []
        for fp in tqdm(mseed_files, desc="加载波形"):
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
                if len(comps) < 3: continue
                z_tr, n_tr, e_tr = comps['Z'], comps['N'], comps['E']
                for tr in [z_tr, n_tr, e_tr]:
                    if tr.stats.sampling_rate != self.target_sr: tr.resample(self.target_sr)
                starts = [tr.stats.starttime for tr in [z_tr, n_tr, e_tr]]
                ends = [tr.stats.endtime for tr in [z_tr, n_tr, e_tr]]
                cs, ce = max(starts), min(ends)
                clen = int((ce - cs) * self.target_sr)
                if clen < self.total_len: continue

                def gd(tr):
                    off = int((cs - tr.stats.starttime) * self.target_sr)
                    return tr.data[off:off + clen]

                self.waveforms.append({'z': gd(z_tr), 'n': gd(n_tr), 'e': gd(e_tr), 'length': clen, 'station': sta})
                if len(self.waveforms) >= max_waveforms: break
            if len(self.waveforms) >= max_waveforms: break

        print(f"加载 {len(self.waveforms)} 个台站长波形")
        np.random.seed(42)
        idx = np.random.permutation(len(self.waveforms))
        sp = int(len(idx) * (1 - val_split))
        self.wf_indices = idx[:sp] if mode == 'train' else idx[sp:]
        print(f"{mode}: {len(self.wf_indices)} 个台站")

    def __len__(self):
        return len(self.wf_indices) * 50

    def __getitem__(self, idx):
        wf = self.waveforms[np.random.choice(self.wf_indices)]
        start = np.random.randint(0, wf['length'] - self.total_len + 1)
        seg = np.stack([wf['z'][start:start + self.total_len],
                        wf['n'][start:start + self.total_len],
                        wf['e'][start:start + self.total_len]]).astype(np.float32)
        for c in range(3):
            seg[c] /= (np.max(np.abs(seg[c])) + 1e-10)
        x = torch.from_numpy(seg[:, :100])
        y = torch.from_numpy(seg[:, 100:400])
        if self.mode == 'train':
            shift = np.random.randint(-3, 4)
            if shift != 0:
                x = torch.roll(x, shift, dims=-1)
                y = torch.roll(y, shift, dims=-1)
            amp = np.random.uniform(0.95, 1.05)
            x *= amp;
            y *= amp
        return x, y


# ==================== 评估 ====================
def correlation_per_component(y_pred, y_true):
    corrs = {}
    for i, comp in enumerate(['Z', 'N', 'E']):
        pred_c = y_pred[:, i, :].reshape(y_pred.shape[0], -1)
        true_c = y_true[:, i, :].reshape(y_true.shape[0], -1)
        pm = pred_c - pred_c.mean(dim=1, keepdim=True)
        tm = true_c - true_c.mean(dim=1, keepdim=True)
        num = (pm * tm).sum(dim=1)
        den = torch.sqrt((pm ** 2).sum(dim=1) * (tm ** 2).sum(dim=1))
        corrs[comp] = (num / (den + 1e-8)).mean().item()
    return corrs


def train_epoch(model, loader, opt, sch, device, epoch):
    model.train()
    met = {'mse': 0, 'corr_z': 0, 'corr_n': 0, 'corr_e': 0}
    pbar = tqdm(loader, desc=f"Train(E{epoch + 1})")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_pred = model(x)
        mse_z = F.mse_loss(y_pred[:, 0, :], y[:, 0, :])
        mse_n = F.mse_loss(y_pred[:, 1, :], y[:, 1, :])
        mse_e = F.mse_loss(y_pred[:, 2, :], y[:, 2, :])
        loss = mse_z + NE_WEIGHT * mse_n + NE_WEIGHT * mse_e
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        met['mse'] += loss.item()
        comp_c = correlation_per_component(y_pred, y)
        met['corr_z'] += comp_c['Z'];
        met['corr_n'] += comp_c['N'];
        met['corr_e'] += comp_c['E']
        pbar.set_postfix(MSE=f"{loss.item():.3f}", Z=f"{comp_c['Z']:.3f}")
    if sch is not None:
        sch.step(epoch)
    return {k: v / len(loader) for k, v in met.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    met = {'mse': 0, 'corr_z': 0, 'corr_n': 0, 'corr_e': 0}
    for x, y in tqdm(loader, desc="Validating"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        mse_z = F.mse_loss(y_pred[:, 0, :], y[:, 0, :])
        mse_n = F.mse_loss(y_pred[:, 1, :], y[:, 1, :])
        mse_e = F.mse_loss(y_pred[:, 2, :], y[:, 2, :])
        met['mse'] += (mse_z + NE_WEIGHT * mse_n + NE_WEIGHT * mse_e).item()
        comp_c = correlation_per_component(y_pred, y)
        met['corr_z'] += comp_c['Z'];
        met['corr_n'] += comp_c['N'];
        met['corr_e'] += comp_c['E']
    return {k: v / len(loader) for k, v in met.items()}


# ==================== PCA拟合 ====================
def fit_pca(model, loader, device, n_coeffs=N_PCA_COEFFS):
    model.eval()
    all_waveforms = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="收集波形拟合PCA"):
            y_pred = model(x.to(device))
            all_waveforms.append(y_pred.reshape(y_pred.shape[0], -1).cpu().numpy())
    all_waveforms = np.concatenate(all_waveforms, axis=0)
    print(f"收集 {all_waveforms.shape[0]} 个波形，每个{all_waveforms.shape[1]}点（三分量拼接）")
    pca = PCA(n_components=n_coeffs)
    pca.fit(all_waveforms)
    print(f"{n_coeffs}个PCA系数恢复能量: {np.sum(pca.explained_variance_ratio_):.2%}")
    return pca


# ==================== 主训练 ====================
def train_predictor(data_dir="./training_data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = MseedRandomDataset(data_dir, 'train', max_waveforms=20000)
    val_ds = MseedRandomDataset(data_dir, 'val', max_waveforms=20000)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = Encoder_Waveform().to(device)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0_WARM, T_mult=2, eta_min=1e-6)

    best_corr = -1
    for epoch in range(EPOCHS):
        train_m = train_epoch(model, train_loader, opt, sch, device, epoch)
        val_m = validate(model, val_loader, device)
        print(
            f"[E{epoch + 1}] Train | Z:{train_m['corr_z']:.3f} N:{train_m['corr_n']:.3f} E:{train_m['corr_e']:.3f} MSE:{train_m['mse']:.4f}")
        print(
            f"[E{epoch + 1}] Val   | Z:{val_m['corr_z']:.3f} N:{val_m['corr_n']:.3f} E:{val_m['corr_e']:.3f} MSE:{val_m['mse']:.4f}")
        if val_m['corr_z'] > best_corr + 0.01:
            best_corr = val_m['corr_z']
            torch.save(model.state_dict(), "predictor_best0.pth")
            print(f"  ✅ 最佳 (Z:{best_corr:.3f})")

    print(f"\n训练完成！最佳 Z_Corr: {best_corr:.3f}")
    model.load_state_dict(torch.load("predictor_best0.pth"))
    pca = fit_pca(model, train_loader, device)
    with open("pca_20coeffSSS.pkl", "wb") as f:
        pickle.dump(pca, f)
    print("PCA已保存到 pca_20coeffSSS.pkl")
    return model, pca


if __name__ == "__main__":
    model, pca = train_predictor()