import os
import sys
import yaml
import torch

from .preprocess import set_seed, TinyCNN, build_fake_loaders
from .train import quick_train
from .evaluate import (
    experiment1_accuracy_efficiency,
    experiment2_robustness,
    experiment3_ablation,
    evaluate_accuracy,
    collect_confusion,
    plot_confusion_matrix,
)
from .preprocess import quantize_model_weights_inplace, find_target_modules, build_iaq4, IAQ4Shim


def main():
    # Load config
    cfg_path = os.path.join('config', 'config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'seed': 123,
            'device': 'auto',
            'dataset': 'fake',
            'image_size': 32,
            'num_classes': 10,
            'calibration': {'n_calib': 128, 'batch_calib': 32},
            'training': {'enable': True, 'steps': 50, 'lr': 1e-3},
            'evaluation': {'batch_eval': 64},
            'iaq4': {'K': 4},
            'output_dir': '.research/iteration1/images'
        }

    out_dir = cfg.get('output_dir', '.research/iteration1/images')
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.get('seed', 123))

    device = cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data (FakeData for a fast functional check)
    calib_loader, train_loader, val_loader = build_fake_loaders(
        n_calib=cfg.get('calibration', {}).get('n_calib', 128),
        n_train=256,
        n_val=256,
        batch_calib=cfg.get('calibration', {}).get('batch_calib', 32),
        batch_eval=cfg.get('evaluation', {}).get('batch_eval', 64),
        num_classes=cfg.get('num_classes', 10),
        img_size=cfg.get('image_size', 32)
    )

    # Model
    model = TinyCNN(in_ch=3, num_classes=cfg.get('num_classes', 10))

    # Quick warm-up training
    if cfg.get('training', {}).get('enable', True):
        model = quick_train(model, train_loader, device=device, iters=cfg.get('training', {}).get('steps', 50), lr=cfg.get('training', {}).get('lr', 1e-3), out_dir=out_dir)

    # Experiment 1
    print('\n==== Running Experiment 1: Accuracy & Efficiency ====')
    res1 = experiment1_accuracy_efficiency(
        model=model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        device=device,
        K=cfg.get('iaq4', {}).get('K', 4),
        num_classes=cfg.get('num_classes', 10),
        out_dir=out_dir
    )

    # Confusion matrix for IAQ-4 variant
    print('\nSaving confusion matrix for IAQ-4 variant...')
    model_iaq = TinyCNN(in_ch=3, num_classes=cfg.get('num_classes', 10))
    model_iaq.load_state_dict(model.state_dict())
    quantize_model_weights_inplace(model_iaq, num_bits=4)
    modules = find_target_modules(model_iaq)
    _, codebooks, luts = build_iaq4(model_iaq, calib_loader, modules, K=cfg.get('iaq4', {}).get('K', 4), device=device)
    iaq_shims = [IAQ4Shim(modules[i], codebooks[i], luts[i]) for i in range(len(modules))]
    cm = collect_confusion(model_iaq, val_loader, num_classes=cfg.get('num_classes', 10), device=device)
    for s in iaq_shims:
        s.remove()
    class_names = [f'C{i}' for i in range(cfg.get('num_classes', 10))]
    plot_confusion_matrix(cm, class_names, out_dir=out_dir, filename='confusion_matrix_iaq4.pdf')

    # Experiment 2
    print('\n==== Running Experiment 2: Robustness ====')
    res2 = experiment2_robustness(
        model=model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        device=device,
        K=cfg.get('iaq4', {}).get('K', 4),
        num_classes=cfg.get('num_classes', 10),
        out_dir=out_dir
    )

    # Experiment 3
    print('\n==== Running Experiment 3: Ablations ====')
    res3 = experiment3_ablation(
        model=model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        device=device,
        Ks=(2, 4, 8),
        fingerprint_bits=(4, 6),
        num_classes=cfg.get('num_classes', 10),
        out_dir=out_dir
    )

    # Print key outputs
    print('\n==== Key Outputs ====')
    print('Experiment 1 accuracies:', res1['acc'])
    print('Experiment 1 overhead model:', res1['overhead_model'])
    print('Experiment 2 corruption means:', res2['mean_acc_corruptions'])
    print('Experiment 3 K ablation:', res3['acc_by_K'])
    print(f'All figures saved under: {out_dir}')


if __name__ == '__main__':
    main()
