import os
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .preprocess import (
    set_seed,
    TinyCNN,
    build_fake_loaders,
    build_corrupted_loader,
    find_target_modules,
    PreActTap,
    quantize_model_weights_inplace,
    IAQ4Shim,
    StaticPTQShim,
    DynamicPercentileShim,
    static_scales_from_calibration,
    build_iaq4,
)


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device='cpu'):
    model.eval().to(device)
    total, correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


@torch.no_grad()
def collect_confusion(model, dataloader, num_classes=10, device='cpu'):
    model.eval().to(device)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        for t, p in zip(labels.view(-1), pred.view(-1)):
            cm[t.long().item(), p.long().item()] += 1
    return cm.cpu().numpy()


def conv2d_macs(module, x_shape):
    Cout = module.out_channels
    Cin = module.in_channels // module.groups
    Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    if isinstance(module.padding, tuple):
        ph, pw = module.padding
    else:
        ph = pw = module.padding
    if isinstance(module.stride, tuple):
        sh, sw = module.stride
    else:
        sh = sw = module.stride
    if isinstance(module.dilation, tuple):
        dh, dw = module.dilation
    else:
        dh = dw = module.dilation
    Hout = (x_shape[2] + 2*ph - dh*(Kh-1) - 1)//sh + 1
    Wout = (x_shape[3] + 2*pw - dw*(Kw-1) - 1)//sw + 1
    return Hout * Wout * Cout * Cin * Kh * Kw


def linear_macs(module, x_shape):
    in_feat = module.in_features
    if len(x_shape) == 2:
        N = x_shape[0]
        return N * in_feat * module.out_features
    else:
        N = int(np.prod(x_shape[:-1]))
        return N * in_feat * module.out_features


def estimate_overhead(modules, example_inputs):
    macs_total = 0
    hist_ops_total = 0
    headers_total_bits = 0
    for mi, m in enumerate(modules):
        xshape = example_inputs.get(mi, None)
        macs = 0
        if xshape is not None:
            if isinstance(m, nn.Conv2d):
                macs = conv2d_macs(m, xshape)
            elif isinstance(m, nn.Linear):
                macs = linear_macs(m, xshape)
        macs_total += macs
        hist_ops_total += 64
        headers_total_bits += 15
    overhead_vs_macs = (hist_ops_total + headers_total_bits) / max(1, macs_total)
    return {
        'macs_total': int(macs_total),
        'hist_ops_total': int(hist_ops_total),
        'meta_bits_total': int(headers_total_bits),
        'overhead_vs_macs': float(overhead_vs_macs)
    }


# Plotting utilities: always save to high-quality PDF in out_dir

def plot_accuracy_bar(results_dict, out_dir, filename='accuracy.pdf'):
    methods = list(results_dict.keys())
    vals = [results_dict[m] for m in methods]
    plt.figure(figsize=(6, 4))
    plt.bar(methods, vals)
    plt.ylabel('Top-1 Accuracy')
    plt.xticks(rotation=30, ha='right')
    plt.title('Accuracy Comparison')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, out_dir, filename='confusion_matrix_iaq4.pdf'):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (IAQ-4)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_gain_curve(gain_to_acc, out_dir, filename='gain_shift_accuracy.pdf'):
    gains = sorted(gain_to_acc.keys())
    vals = [gain_to_acc[g] for g in gains]
    plt.figure(figsize=(5, 4))
    plt.plot(gains, vals, marker='o')
    plt.xlabel('Gain g in (1+g)·x')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Gain Shift Robustness')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_bits_per_act(bits_per_act_by_bs, out_dir, filename='metadata_bits_per_activation.pdf'):
    bss = sorted(bits_per_act_by_bs.keys())
    vals = [bits_per_act_by_bs[b] for b in bss]
    plt.figure(figsize=(5, 4))
    plt.plot(bss, vals, marker='s')
    plt.xlabel('Batch Size')
    plt.ylabel('Avg Metadata bits/activation')
    plt.title('Header Amortization')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_ablation_curve(xvals, yvals, xlabel, title, out_dir, filename):
    plt.figure(figsize=(5, 4))
    plt.plot(xvals, yvals, marker='^')
    plt.xlabel(xlabel)
    plt.ylabel('Top-1 Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def plot_latency(baseline_s, iaq_s, out_dir, filename='inference_latency.pdf'):
    plt.figure(figsize=(5, 4))
    methods = ['FP32/no-shim', 'IAQ-4']
    vals = [baseline_s, iaq_s]
    plt.bar(methods, vals, color=['gray', 'orange'])
    plt.ylabel('Seconds per Evaluation (wall-clock)')
    plt.title('Latency Overhead (Python)')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close()


def experiment1_accuracy_efficiency(model, calib_loader, val_loader, device='cpu', K=4, num_classes=10, weight_bits=4, act_bits=4, out_dir='.research/iteration4/images'):
    print('Experiment 1: Building target modules...')
    modules = find_target_modules(model)
    print(f'Found {len(modules)} modules to quantize (Conv/Linear).')

    # Example input shapes for overhead estimate
    taps = [PreActTap(m) for m in modules]
    with torch.no_grad():
        images, _ = next(iter(calib_loader))
        images = images.to(device)
        _ = model.to(device)(images)
    example_inputs = {}
    for mi, t in enumerate(taps):
        if t.last_inp is not None:
            example_inputs[mi] = tuple(t.last_inp.shape)
        t.remove()

    # Baseline FP32 accuracy and time
    t0 = time.time()
    acc_fp32 = evaluate_accuracy(model, val_loader, device=device)
    t_fp32 = time.time() - t0

    # Quantize weights statically for all 4-bit variants
    model_s4 = copy.deepcopy(model)
    quantize_model_weights_inplace(model_s4, num_bits=weight_bits)
    modules_s4 = find_target_modules(model_s4)

    # Static 4-bit (minmax)
    scales_minmax, unsigned_minmax = static_scales_from_calibration(model_s4, calib_loader, modules_s4, device=device, mode='minmax', num_bits=act_bits)
    shims_minmax = [StaticPTQShim(modules_s4[i], scales_minmax[i], unsigned_minmax[i], num_bits=act_bits) for i in range(len(modules_s4))]
    t0 = time.time()
    acc_s4_minmax = evaluate_accuracy(model_s4, val_loader, device=device)
    t_s4_minmax = time.time() - t0
    for s in shims_minmax:
        s.remove()

    # Static 4-bit (p99)
    scales_p99, unsigned_p99 = static_scales_from_calibration(model_s4, calib_loader, modules_s4, device=device, mode='p99', num_bits=act_bits)
    shims_p99 = [StaticPTQShim(modules_s4[i], scales_p99[i], unsigned_p99[i], num_bits=act_bits) for i in range(len(modules_s4))]
    t0 = time.time()
    acc_s4_p99 = evaluate_accuracy(model_s4, val_loader, device=device)
    t_s4_p99 = time.time() - t0
    for s in shims_p99:
        s.remove()

    # Dynamic Percentile (baseline)
    dyn_shims = [DynamicPercentileShim(modules_s4[i], unsigned=unsigned_p99[i], num_bits=act_bits) for i in range(len(modules_s4))]
    t0 = time.time()
    acc_dyn = evaluate_accuracy(model_s4, val_loader, device=device)
    t_dyn = time.time() - t0
    for s in dyn_shims:
        s.remove()

    # IAQ-4: build codebooks and LUTs from calibration
    print('Experiment 1: Calibrating IAQ-4 (histograms/codebooks/LUTs)...')
    per_mod, codebooks, luts = build_iaq4(model_s4, calib_loader, modules_s4, K=K, device=device, max_samples_per_mod=24)

    # Install IAQ-4 shims
    iaq_shims = [IAQ4Shim(modules_s4[i], codebooks[i], luts[i]) for i in range(len(modules_s4))]
    t0 = time.time()
    acc_iaq = evaluate_accuracy(model_s4, val_loader, device=device)
    t_iaq = time.time() - t0
    for s in iaq_shims:
        s.remove()

    # Overhead estimates
    overhead = estimate_overhead(modules, example_inputs)

    print('Experiment 1 summary:')
    print({'acc_fp32': acc_fp32,
           'acc_s4_minmax': acc_s4_minmax,
           'acc_s4_p99': acc_s4_p99,
           'acc_dyn_percentile': acc_dyn,
           'acc_iaq4': acc_iaq,
           'overhead_model': overhead,
           'latency_wallclock_s': {'fp32': t_fp32, 's4_minmax': t_s4_minmax, 's4_p99': t_s4_p99, 'dyn': t_dyn, 'iaq4': t_iaq}})

    # Plots (PDF)
    results = {
        'FP32': acc_fp32,
        'S4_minmax': acc_s4_minmax,
        'S4_p99': acc_s4_p99,
        'DynamicPercentile': acc_dyn,
        'IAQ-4': acc_iaq
    }
    plot_accuracy_bar(results, out_dir=out_dir, filename='accuracy.pdf')
    plot_latency(t_fp32, t_iaq, out_dir=out_dir, filename='inference_latency.pdf')

    # Bits/activation across batch sizes
    bits_by_bs = {}
    for B in [1, 8, 32, 128]:
        vals = []
        for mi in example_inputs:
            n_acts = int(np.prod((B,) + example_inputs[mi][1:]))
            vals.append(15.0 / max(1, n_acts))
        bits_by_bs[B] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    plot_bits_per_act(bits_by_bs, out_dir=out_dir, filename='metadata_bits_per_activation.pdf')

    return {
        'acc': results,
        'overhead_model': overhead,
        'latency_wallclock_s': {'fp32': t_fp32, 'iaq4': t_iaq},
        'bits_by_bs': bits_by_bs,
        'modules': modules_s4,
        'codebooks': codebooks,
        'luts': luts
    }


def experiment2_robustness(model, calib_loader, val_loader, device='cpu', K=4, num_classes=10, act_bits=4, out_dir='.research/iteration4/images'):
    modules = find_target_modules(model)
    # Weight quantization for 4-bit flows
    model_q = copy.deepcopy(model)
    quantize_model_weights_inplace(model_q, num_bits=4)
    modules_q = find_target_modules(model_q)

    # IAQ-4 build
    print('Experiment 2: Calibrating IAQ-4...')
    _, codebooks, luts = build_iaq4(model_q, calib_loader, modules_q, K=K, device=device)
    iaq_shims = [IAQ4Shim(modules_q[i], codebooks[i], luts[i]) for i in range(len(modules_q))]

    # Corruptions: gaussian_noise, blur, brightness (severities 1..5)
    base_ds = val_loader.dataset
    corr_types = ['gaussian_noise', 'blur', 'brightness']
    mean_acc = {}
    for corr in corr_types:
        accs = []
        for sev in [1, 2, 3, 4, 5]:
            loader_c = build_corrupted_loader(base_ds, corruption=corr, severity=sev, batch_size=val_loader.batch_size)
            acc = evaluate_accuracy(model_q, loader_c, device=device)
            accs.append(acc)
        mean_acc[corr] = float(np.mean(accs))
        print(f'Robustness ({corr}): per-severity={accs}, mean={mean_acc[corr]:.4f}')

    # Gain shift robustness
    gains = [-0.2, -0.1, 0.0, 0.1, 0.2]
    gain_to_acc = {}
    for g in gains:
        def scale_tf(img):
            x = img if torch.is_tensor(img) else transforms.ToTensor()(img)
            x = torch.clamp(x * (1.0 + g), 0.0, 1.0)
            return x
        class GainWrapper(torch.utils.data.Dataset):
            def __init__(self, ds, tf):
                self.ds, self.tf = ds, tf
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                img, y = self.ds[idx]
                return self.tf(img), y
        loader_g = DataLoader(GainWrapper(base_ds, scale_tf), batch_size=val_loader.batch_size, shuffle=False)
        accg = evaluate_accuracy(model_q, loader_g, device=device)
        gain_to_acc[g] = accg
    print('Gain shift accuracy:', gain_to_acc)

    plot_gain_curve(gain_to_acc, out_dir=out_dir, filename='gain_shift_accuracy.pdf')

    for s in iaq_shims:
        s.remove()

    return {'mean_acc_corruptions': mean_acc, 'gain_to_acc': gain_to_acc}


def experiment3_ablation(model, calib_loader, val_loader, device='cpu', Ks=(2, 4, 8), fingerprint_bits=(4, 6), num_classes=10, out_dir='.research/iteration4/images'):
    # Weight quantization
    model_q = copy.deepcopy(model)
    quantize_model_weights_inplace(model_q, num_bits=4)
    modules = find_target_modules(model_q)

    acc_by_K = {}
    for K in Ks:
        _, codebooks, luts = build_iaq4(model_q, calib_loader, modules, K=K, device=device)
        shims = [IAQ4Shim(modules[i], codebooks[i], luts[i]) for i in range(len(modules))]
        acc = evaluate_accuracy(model_q, val_loader, device=device)
        acc_by_K[K] = acc
        for s in shims:
            s.remove()
        print(f'Ablation: K={K}, accuracy={acc:.4f}')
    plot_ablation_curve(list(acc_by_K.keys()), list(acc_by_K.values()), xlabel='Codebook size K', title='Ablation: K vs Accuracy', out_dir=out_dir, filename='ablation_K_accuracy.pdf')

    # Fingerprint bits ablation: compress LUT 64->16 entries by merging by MSBs when fb=4
    acc_by_fpbits = {}
    for fb in fingerprint_bits:
        if fb == 6:
            _, codebooks, luts = build_iaq4(model_q, calib_loader, modules, K=4, device=device)
        else:
            _, codebooks, luts = build_iaq4(model_q, calib_loader, modules, K=4, device=device)
            new_luts = {}
            for mi in luts:
                lut = np.array(luts[mi])
                group = 2 ** (6 - fb)
                comp = []
                for i in range(0, 64, group):
                    seg = lut[i:i + group]
                    vals, counts = np.unique(seg, return_counts=True)
                    comp.append(int(vals[np.argmax(counts)]))
                expanded = np.repeat(np.array(comp), group)
                new_luts[mi] = expanded.tolist()
            luts = new_luts
        shims = [IAQ4Shim(modules[i], codebooks[i], luts[i]) for i in range(len(modules))]
        acc = evaluate_accuracy(model_q, val_loader, device=device)
        acc_by_fpbits[fb] = acc
        for s in shims:
            s.remove()
        print(f'Ablation: fingerprint_bits={fb}, accuracy={acc:.4f}')
    plot_ablation_curve(list(acc_by_fpbits.keys()), list(acc_by_fpbits.values()), xlabel='Fingerprint bits', title='Ablation: Fingerprint bits vs Accuracy', out_dir=out_dir, filename='ablation_fingerprint_bits_accuracy.pdf')

    return {'acc_by_K': acc_by_K, 'acc_by_fingerprint_bits': acc_by_fpbits}
