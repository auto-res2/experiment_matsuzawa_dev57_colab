import os
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def build_fake_loaders(n_calib=128, n_train=256, n_val=256, batch_calib=32, batch_eval=64, num_classes=10, img_size=32):
    tf = transforms.Compose([transforms.ToTensor()])
    calib_ds = datasets.FakeData(size=n_calib, image_size=(3, img_size, img_size), num_classes=num_classes, transform=tf)
    train_ds = datasets.FakeData(size=n_train, image_size=(3, img_size, img_size), num_classes=num_classes, transform=tf)
    val_ds = datasets.FakeData(size=n_val, image_size=(3, img_size, img_size), num_classes=num_classes, transform=tf)
    calib_loader = DataLoader(calib_ds, batch_size=batch_calib, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False)
    return calib_loader, train_loader, val_loader


def build_corrupted_loader(base_dataset, corruption='gaussian_noise', severity=1, batch_size=64):
    s = max(1, min(5, severity))
    if corruption == 'gaussian_noise':
        std = 0.05 * s
        def tf(img):
            x = transforms.ToTensor()(img)
            noise = torch.randn_like(x) * std
            return torch.clamp(x + noise, 0.0, 1.0)
    elif corruption == 'blur':
        k = 2 * s + 1
        tf_blur = transforms.GaussianBlur(kernel_size=k, sigma=s * 0.5)
        def tf(img):
            x = tf_blur(img)
            return transforms.ToTensor()(x)
    elif corruption == 'brightness':
        factor = 1.0 + 0.15 * s
        tf_jit = transforms.ColorJitter(brightness=factor)
        def tf(img):
            x = tf_jit(img)
            return transforms.ToTensor()(x)
    else:
        def tf(img):
            return transforms.ToTensor()(img)

    class CorruptWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, tf):
            self.ds = ds
            self.tf = tf
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, y = self.ds[idx]
            return self.tf(transforms.ToPILImage()(img) if torch.is_tensor(img) else img), y

    wrapped = CorruptWrapper(base_dataset, tf)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=False)


def quantize_dequant(x, scale, num_bits=4, unsigned=False):
    scale = torch.clamp(torch.as_tensor(scale, dtype=x.dtype, device=x.device), min=1e-12)
    if unsigned:
        qmin, qmax = 0, (2 ** num_bits) - 1
        zp = 0.0
    else:
        qmin, qmax = -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1
        zp = 0.0
    y = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
    return (y - zp) * scale


def quantize_model_weights_inplace(model, num_bits=4):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                w = m.weight.data
                maxv = float(w.abs().max().cpu())
                if maxv < 1e-12:
                    continue
                qmax = (2 ** (num_bits - 1)) - 1
                scale = maxv / qmax
                qw = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
                m.weight.data = qw * scale
    return model


def _log_edges_64(xmin, xmax):
    xmin = max(xmin, 1e-9)
    xmax = max(xmax, xmin * 10)
    return torch.logspace(math.log10(xmin), math.log10(xmax), steps=65)


def abs_hist64_and_stats(x):
    xa = x.detach().abs().flatten()
    xa = xa[torch.isfinite(xa)]
    if xa.numel() == 0:
        h = torch.zeros(64, device='cpu')
        return h, 0.0, 0.0
    xmin = float(torch.clamp_min(xa[xa > 0].min(), 1e-9)) if (xa > 0).any() else 1e-9
    xmax = float(xa.max())
    edges = _log_edges_64(xmin, xmax)
    h = torch.histc(xa, bins=64, min=float(edges[0]), max=float(edges[-1]))
    cdf = torch.cumsum(h, dim=0)
    total = float(cdf[-1].item() + 1e-9)
    target = 0.99 * total
    bin_idx = int(torch.searchsorted(cdf, torch.tensor(target, device=cdf.device)))
    bin_idx = max(0, min(63, bin_idx))
    p99 = float(edges[bin_idx + 1].cpu())
    amean = float(xa.mean().cpu())
    h = (h / (h.sum() + 1e-9)).cpu()
    return h, amean, p99


def logbin_scalar(x, xmin=1e-6, xmax=20.0, bits=3):
    x = float(x)
    x = min(max(x, xmin), xmax)
    levels = 2 ** bits
    idx = int(round((math.log(x) - math.log(xmin)) / (math.log(xmax) - math.log(xmin)) * (levels - 1)))
    return max(0, min(levels - 1, idx))


def fingerprint6(abs_mean, p99):
    a = logbin_scalar(abs_mean, xmin=1e-6, xmax=10.0, bits=3)
    b = logbin_scalar(p99, xmin=1e-6, xmax=20.0, bits=3)
    return (a << 3) | b


class PreActTap:
    def __init__(self, module):
        self.module = module
        self.last_inp = None
        self.hook = module.register_forward_pre_hook(self._hook)
    def _hook(self, m, inp):
        with torch.no_grad():
            self.last_inp = inp[0].detach()
    def remove(self):
        self.hook.remove()


def find_target_modules(model):
    targets = []
    for _, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            targets.append(m)
    return targets


def kmeans_torch(data_np, K=4, iters=20, seed=123):
    torch.manual_seed(seed)
    data = torch.from_numpy(np.asarray(data_np, dtype=np.float32))
    N, D = data.shape
    K = min(K, max(1, N))
    idx = torch.randperm(N)[:K]
    centroids = data[idx].clone()
    for _ in range(iters):
        d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=2)
        assign = torch.argmin(d2, dim=1)
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                centroids[k] = data[mask].mean(dim=0)
    return assign.numpy(), centroids.numpy()


def collect_calibration(model, calib_loader, modules, device='cpu', max_samples_per_mod=24):
    taps = [PreActTap(m) for m in modules]
    model.eval().to(device)
    per_mod = {i: {'hists': [], 'abs_mean': [], 'p99': [], 'samples': []} for i in range(len(modules))}

    it = calib_loader
    if TQDM:
        try:
            it = tqdm(calib_loader, desc='Calibrating', leave=False)
        except Exception:
            it = calib_loader

    with torch.no_grad():
        for images, _ in it:
            images = images.to(device)
            _ = model(images)
            for mi, tap in enumerate(taps):
                x = tap.last_inp
                if x is None:
                    continue
                h, am, p = abs_hist64_and_stats(x)
                per_mod[mi]['hists'].append(h.numpy())
                per_mod[mi]['abs_mean'].append(am)
                per_mod[mi]['p99'].append(p)
                if len(per_mod[mi]['samples']) < max_samples_per_mod:
                    xs = x.detach().float().cpu()
                    if xs.dim() == 4:
                        c = min(xs.size(1), 8)
                        xs = xs[:1, :c, :min(14, xs.size(2)), :min(14, xs.size(3))].contiguous()
                    elif xs.dim() == 2:
                        xs = xs[:1, :min(512, xs.size(1))]
                    per_mod[mi]['samples'].append(xs)

    for t in taps:
        t.remove()

    for mi in per_mod:
        if len(per_mod[mi]['hists']) == 0:
            per_mod[mi]['hists'] = np.zeros((1, 64), dtype=np.float32)
        else:
            per_mod[mi]['hists'] = np.stack(per_mod[mi]['hists'], axis=0)
    return per_mod


def detect_unsigned(per_mod_entry):
    mins = []
    for xs in per_mod_entry['samples']:
        mins.append(float(xs.min()))
    return (len(mins) > 0) and (min(mins) >= -1e-6)


def choose_clip_grid(anchor, num_candidates=40, low=0.6, high=1.4):
    low = max(0.1, low)
    return np.linspace(low * max(anchor, 1e-6), high * max(anchor, 1e-6), num_candidates)


def build_codebook_for_module(model, module, mod_idx, per_mod_entry, K=4, device='cpu'):
    H = per_mod_entry['hists']
    if H.shape[0] < K:
        K = max(1, H.shape[0])
    labels, centroids = kmeans_torch(H, K=K, iters=20, seed=123)

    is_unsigned = detect_unsigned(per_mod_entry)
    batches_by_centroid = defaultdict(list)
    for bidx, lab in enumerate(labels):
        batches_by_centroid[lab].append(bidx)
    p99s = per_mod_entry['p99']

    codebook = []
    module = module.to(device)
    module.eval()

    def run_module(m, x):
        return m(x)

    for k in range(K):
        members = batches_by_centroid.get(k, [])
        if len(p99s) > 0:
            if len(members) > 0:
                sel = [p99s[i] for i in members if i < len(p99s)]
                anchor = float(np.median(sel)) if len(sel) > 0 else float(np.median(p99s))
            else:
                anchor = float(np.median(p99s))
        else:
            anchor = 1.0
        grid = choose_clip_grid(anchor, num_candidates=40, low=0.6, high=1.4)

        xs_list = per_mod_entry['samples'][:12] if len(per_mod_entry['samples']) > 0 else []
        if len(xs_list) == 0:
            qmax = 2 ** 4 - 1 if is_unsigned else 2 ** 3 - 1
            scale = anchor / max(qmax, 1)
            codebook.append({'scale': float(scale), 'unsigned': is_unsigned})
            continue

        xs_list_dev = [xs.to(device) for xs in xs_list]
        with torch.no_grad():
            y_fp = [run_module(module, xs) for xs in xs_list_dev]

        best_mse, best_scale = float('inf'), None
        qmax = (2 ** 4 - 1) if is_unsigned else ((2 ** 3) - 1)
        for clip in grid:
            scale = clip / max(qmax, 1)
            mses = []
            with torch.no_grad():
                for xs, y in zip(xs_list_dev, y_fp):
                    xq = quantize_dequant(xs, scale, num_bits=4, unsigned=is_unsigned)
                    yq = run_module(module, xq)
                    mses.append(F.mse_loss(yq, y).item())
            mse = float(np.mean(mses)) if len(mses) > 0 else 1e9
            if mse < best_mse:
                best_mse, best_scale = mse, float(scale)
        codebook.append({'scale': float(best_scale), 'unsigned': is_unsigned})

    votes = np.zeros((64, K), dtype=np.int32)
    for b in range(H.shape[0]):
        am_list = per_mod_entry['abs_mean']
        p99_list = per_mod_entry['p99']
        am = am_list[b] if b < len(am_list) else (am_list[-1] if len(am_list) > 0 else 1.0)
        p = p99_list[b] if b < len(p99_list) else (p99_list[-1] if len(p99_list) > 0 else 1.0)
        fp = fingerprint6(am, p)
        lab = labels[b]
        votes[fp, lab] += 1
    lut = votes.argmax(axis=1)
    empty = (votes.sum(axis=1) == 0)
    if empty.any():
        filled = np.where(~empty)[0]
        if len(filled) == 0:
            lut[:] = 0
        else:
            for fp in np.where(empty)[0]:
                best_cand, best_dist, best_cnt = None, 999, -1
                for cand in filled:
                    dist = bin(fp ^ cand).count('1')
                    cnt = votes[cand].sum()
                    if (dist < best_dist) or (dist == best_dist and cnt > best_cnt):
                        best_cand, best_dist, best_cnt = cand, dist, cnt
                lut[fp] = lut[best_cand]

    return codebook, lut.tolist()


def build_iaq4(model, calib_loader, modules, K=4, device='cpu', max_samples_per_mod=24):
    per_mod = collect_calibration(model, calib_loader, modules, device=device, max_samples_per_mod=max_samples_per_mod)
    codebooks, luts = {}, {}
    it = range(len(modules))
    for mi in it:
        cb, lut = build_codebook_for_module(model, modules[mi], mi, per_mod[mi], K=K, device=device)
        codebooks[mi] = cb
        luts[mi] = lut
    return per_mod, codebooks, luts


class IAQ4Shim:
    def __init__(self, module, codebook, lut):
        self.module = module
        self.codebook = codebook
        self.lut = torch.tensor(lut, dtype=torch.long)
        self.last_scale = None
        self.last_unsigned = None
        self.meta_headers = 0
        self._hook = module.register_forward_pre_hook(self._hook_fn)
    def remove(self):
        self._hook.remove()
    def _hook_fn(self, m, inp):
        with torch.no_grad():
            x = inp[0]
            _, am, p99 = abs_hist64_and_stats(x)
            fp = fingerprint6(am, p99)
            idx = int(self.lut[fp].item())
            idx = min(idx, len(self.codebook) - 1)
            scale = self.codebook[idx]['scale']
            unsigned = self.codebook[idx]['unsigned']
            self.last_scale, self.last_unsigned = scale, unsigned
            self.meta_headers += 1
            xq = quantize_dequant(x, scale, num_bits=4, unsigned=unsigned)
        return (xq,)


class StaticPTQShim:
    def __init__(self, module, scale, unsigned=False, num_bits=4):
        self.module = module
        self.scale = float(scale)
        self.unsigned = bool(unsigned)
        self.num_bits = num_bits
        self._hook = module.register_forward_pre_hook(self._hook_fn)
    def remove(self):
        self._hook.remove()
    def _hook_fn(self, m, inp):
        x = inp[0]
        return (quantize_dequant(x, self.scale, num_bits=self.num_bits, unsigned=self.unsigned),)


class DynamicPercentileShim:
    def __init__(self, module, unsigned=False, num_bits=4):
        self.module = module
        self.unsigned = bool(unsigned)
        self.num_bits = num_bits
        self._hook = module.register_forward_pre_hook(self._hook_fn)
    def remove(self):
        self._hook.remove()
    def _hook_fn(self, m, inp):
        x = inp[0]
        _, _, p99 = abs_hist64_and_stats(x)
        qmax = (2 ** self.num_bits - 1) if self.unsigned else (2 ** (self.num_bits - 1) - 1)
        scale = p99 / max(qmax, 1)
        return (quantize_dequant(x, scale, num_bits=self.num_bits, unsigned=self.unsigned),)


def static_scales_from_calibration(model, calib_loader, modules, device='cpu', mode='minmax', num_bits=4):
    taps = [PreActTap(m) for m in modules]
    model.eval().to(device)
    mins = defaultdict(lambda: float('inf'))
    maxs = defaultdict(lambda: float('-inf'))
    p99s = defaultdict(list)

    it = calib_loader
    if TQDM:
        try:
            it = tqdm(calib_loader, desc='Static scales', leave=False)
        except Exception:
            it = calib_loader

    with torch.no_grad():
        for images, _ in it:
            images = images.to(device)
            _ = model(images)
            for mi, tap in enumerate(taps):
                x = tap.last_inp
                if x is None:
                    continue
                mins[mi] = min(mins[mi], float(x.min()))
                maxs[mi] = max(maxs[mi], float(x.max()))
                _, _, p = abs_hist64_and_stats(x)
                p99s[mi].append(p)

    for t in taps:
        t.remove()

    scales, unsigned = {}, {}
    for mi in range(len(modules)):
        mn, mx = mins[mi], maxs[mi]
        is_unsigned = (mn >= -1e-6)
        unsigned[mi] = is_unsigned
        if mode == 'minmax':
            if is_unsigned:
                clip = mx
                qmax = 2 ** num_bits - 1
            else:
                clip = max(abs(mn), abs(mx))
                qmax = 2 ** (num_bits - 1) - 1
        else:
            p = np.median(p99s[mi]) if len(p99s[mi]) > 0 else max(abs(mn), abs(mx))
            clip = p
            qmax = (2 ** num_bits - 1) if is_unsigned else (2 ** (num_bits - 1) - 1)
        scale = clip / max(qmax, 1)
        scales[mi] = float(scale)
    return scales, unsigned
