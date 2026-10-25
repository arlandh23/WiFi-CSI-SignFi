import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"))


def _grad_scaler(device):
    try:
        return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, clip=1.0):
    train = optimizer is not None
    model.train(train)
    tot, preds, trues = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), _autocast(device):
            out = model(x)
            loss = criterion(out.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * x.size(0)
        preds.append(out.argmax(1).cpu()); trues.append(y.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return tot / len(t), accuracy_score(t, p), f1_score(t, p, average="macro")


def fit(model, loaders, device, epochs=40, lr=1e-3, wd=1e-4, patience=10, label_smoothing=0.05,
        ckpt_path=None, verbose=True):
    model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = _grad_scaler(device)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]}
    best, best_state, best_ep, wait = (-1, np.inf), None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta, _ = run_epoch(model, loaders["train"], crit, device, opt, scaler)
        vl, va, vf = run_epoch(model, loaders["val"], crit, device)
        hist["lr"].append(opt.param_groups[0]["lr"]); sched.step()
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"], [tl, ta, vl, va, vf]):
            hist[k].append(v)
        improved = (va > best[0]) or (va == best[0] and vl < best[1])
        if improved:
            best, best_ep, wait = (va, vl), ep, 0
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            wait += 1
        if verbose and (ep == 1 or ep % 5 == 0 or improved):
            print(f"ep {ep:3d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} f1 {vf:.3f}"
                  + ("  *" if improved else ""))
        if wait >= patience:
            if verbose:
                print(f"Early stopping di epoch {ep} (best epoch {best_ep})")
            break
    model.load_state_dict(best_state)
    return model, hist, {"best_epoch": best_ep, "best_val_acc": best[0], "train_time_s": time.time() - t0}


@torch.no_grad()
def predict(model, loader, device):
    model.eval().to(device)
    P, Y = [], []
    for x, y in loader:
        with _autocast(device):
            P.append(torch.softmax(model(x.to(device)).float(), 1).cpu())
        Y.append(y)
    probs = torch.cat(P).numpy()
    return torch.cat(Y).numpy(), probs.argmax(1), probs


@torch.no_grad()
def latency_ms(model, sample, device, n=50):
    model.eval().to(device); x = sample.to(device)
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n / x.shape[0] * 1000
