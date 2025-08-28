import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def quick_train(model, train_loader, device='cpu', iters=50, lr=1e-3, out_dir='.research/iteration2/images'):
    """A short warm-up training run to make FakeData non-degenerate.
    Saves training loss curve as a high-quality PDF.
    """
    model.train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    steps = 0
    it = iter(train_loader)
    t0 = time.time()
    while steps < iters:
        try:
            images, labels = next(it)
        except StopIteration:
            it = iter(train_loader)
            images, labels = next(it)
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        steps += 1
    t1 = time.time()

    # Save training loss curve (PDF)
    plt.figure(figsize=(5, 4))
    plt.plot(losses, linewidth=2.0)
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.title('Quick Training Loss ({} steps, {:.2f}s)'.format(iters, t1 - t0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_loss_baseline.pdf", bbox_inches='tight')
    plt.close()

    return model
