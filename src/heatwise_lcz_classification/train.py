#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train/evaluate LCZ_HMSSNet on one or more patch H5 files.

Usage:
    python train.py --h5-dir /path/to/Patch/Berlin_block500.h5 \
                     --output-dir /path/to/results/Berlin \
                     --config config/train_config.example.yaml

`--h5-dir` may be a single .h5 file or a directory containing several; all are
concatenated. LST/PCA fields and a predefined 'split' dataset are used
automatically if present in the H5 (see heatwise-patch-extraction), otherwise
falling back to a random stratified split.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import LCZ_HMSSNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def collect_h5_files(h5_dir: str) -> list[str]:
    if os.path.isfile(h5_dir) and h5_dir.endswith(".h5"):
        return [h5_dir]
    files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 files found at {h5_dir}")
    return files


def read_optional_split(f):
    if "split" not in f:
        return None
    split = f["split"][:]
    if split.dtype.kind in {"S", "O", "U"}:
        split = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in split])
    return split


def split_indices_from_dataset(split_values):
    if split_values is None:
        return None
    if split_values.dtype.kind in {"i", "u"}:
        idx_train = np.where(split_values == 0)[0]
        idx_val = np.where(split_values == 1)[0]
        idx_test = np.where(split_values == 2)[0]
    else:
        lowered = np.char.lower(split_values.astype(str))
        idx_train = np.where(lowered == "train")[0]
        idx_val = np.where(lowered == "val")[0]
        idx_test = np.where(lowered == "test")[0]
    if min(len(idx_train), len(idx_val), len(idx_test)) == 0:
        raise SystemExit("H5 split exists, but train/val/test are not all present.")
    return idx_train, idx_val, idx_test


def load_dataset(h5_files: list[str], sen2_scale: float):
    sen2_list, hsi_bs_list, hsi_pca_list, lbl_list, lst_list, split_list = [], [], [], [], [], []
    all_have_pca = all_have_lst = all_have_split = True

    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            s = f["sen2"][:].astype(np.float32) / sen2_scale
            hb = f["hsi_bs"][:].astype(np.float32)
            lb = np.argmax(f["label"][:], axis=1).astype(np.int64)

            if "hsi_pca" in f:
                hsi_pca_list.append(f["hsi_pca"][:].astype(np.float32))
                pca_info = f"  hsi_pca{hsi_pca_list[-1].shape[1:]}"
            else:
                all_have_pca = False
                pca_info = "  hsi_pca<MISSING>"

            if "lst" in f:
                lt = f["lst"][:].astype(np.float32)
                if lt.ndim == 3:
                    lt = lt[..., None]
                lst_list.append(lt)
                lst_info = f"  lst{lt.shape[1:]}"
            else:
                all_have_lst = False
                lst_info = "  lst<MISSING>"

            split = read_optional_split(f)
            if split is None:
                all_have_split = False
            else:
                split_list.append(split)

        print(f"  {os.path.basename(fp)}: N={len(lb)}  sen2{s.shape[1:]}  hsi_bs{hb.shape[1:]}{pca_info}{lst_info}")
        sen2_list.append(s)
        hsi_bs_list.append(hb)
        lbl_list.append(lb)

    for name, arrays in [("sen2", sen2_list), ("hsi_bs", hsi_bs_list)]:
        shapes = {a.shape[1:] for a in arrays}
        if len(shapes) != 1:
            raise SystemExit(f"Dataset {name} has inconsistent shapes across H5 files: {shapes}")

    hsi_pca = None
    if all_have_pca:
        shapes = {a.shape[1:] for a in hsi_pca_list}
        if len(shapes) != 1:
            raise SystemExit(f"Dataset hsi_pca has inconsistent shapes across H5 files: {shapes}")
        hsi_pca = np.concatenate(hsi_pca_list, axis=0)

    lst = None
    if all_have_lst:
        shapes = {a.shape[1:] for a in lst_list}
        if len(shapes) != 1:
            raise SystemExit(f"Dataset lst has inconsistent shapes across H5 files: {shapes}")
        lst = np.concatenate(lst_list, axis=0)

    sen2 = np.concatenate(sen2_list, axis=0)
    hsi_bs = np.concatenate(hsi_bs_list, axis=0)
    lbl = np.concatenate(lbl_list, axis=0)
    split_values = np.concatenate(split_list, axis=0) if all_have_split else None

    mask = ~np.isnan(sen2).any(axis=(1, 2, 3)) & ~np.isnan(hsi_bs).any(axis=(1, 2, 3))
    if hsi_pca is not None:
        mask &= ~np.isnan(hsi_pca).any(axis=(1, 2, 3))
    if lst is not None:
        mask &= ~np.isnan(lst).any(axis=(1, 2, 3))

    sen2, hsi_bs, lbl = sen2[mask], hsi_bs[mask], lbl[mask]
    if hsi_pca is not None:
        hsi_pca = hsi_pca[mask]
    if lst is not None:
        lst = lst[mask]
    if split_values is not None:
        split_values = split_values[mask]

    print(f"\nMerged {len(h5_files)} H5 files -> N={len(lbl)} | sen2{sen2.shape[1:]}  hsi_bs{hsi_bs.shape[1:]}"
          + (f"  hsi_pca{hsi_pca.shape[1:]}" if hsi_pca is not None else "  hsi_pca<MISSING>")
          + (f"  lst{lst.shape[1:]}" if lst is not None else "  lst<MISSING>"))
    print("Class distribution:", dict(zip(*np.unique(lbl, return_counts=True))))

    return sen2, hsi_bs, hsi_pca, lst, lbl, split_values


class PatchDataset(Dataset):
    def __init__(self, X1=None, X2=None, X3=None, labels=None, augment=False):
        arrays = [x for x in (X1, X2, X3) if x is not None]
        if not arrays:
            raise ValueError("At least one input array is required.")
        X = np.concatenate(arrays, axis=-1) if len(arrays) > 1 else arrays[0]

        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).unsqueeze(1).float()
        self.y = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _aug(self, x):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[-2, -1])
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._aug(x)
        return x, self.y[idx]


class EarlyStopping:
    def __init__(self, patience=10, path="best_model.pth"):
        self.patience = patience
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_model(model, train_loader, val_loader, max_epochs, es=None):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"[{epoch:03d}/{max_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

        if es:
            es(epoch_val_loss, model)
            if es.stop:
                print(f"Early stopping at epoch {epoch}")
                break

    if es:
        model.load_state_dict(torch.load(es.path, map_location=DEVICE))


def evaluate_model(model, test_loader, output_root, num_classes, mode=""):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    oa = accuracy_score(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    labels = np.arange(num_classes)
    cm = confusion_matrix(trues, preds, labels=labels)

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = np.divide(per_class_correct, per_class_total,
                               out=np.zeros_like(per_class_correct, dtype=float),
                               where=per_class_total > 0)

    print(f"\n{mode} Results:")
    print(f"  OA    = {oa:.4f}")
    print(f"  Kappa = {kappa:.4f}")
    valid_mask = per_class_total > 0
    macro_recall = per_class_acc[valid_mask].mean() if valid_mask.any() else np.nan
    print(f"  Macro recall = {macro_recall*100:.2f}%")

    safe_mode = clean_name(mode)
    np.save(os.path.join(output_root, f"confusion_matrix_{safe_mode}.npy"), cm)
    pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]
                 ).to_csv(os.path.join(output_root, f"confusion_matrix_{safe_mode}.csv"))
    pd.DataFrame({
        "class_id": labels, "support": per_class_total,
        "correct": per_class_correct, "per_class_accuracy": per_class_acc,
    }).to_csv(os.path.join(output_root, f"per_class_accuracy_{safe_mode}.csv"), index=False)

    cm_norm = cm.astype("float") / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[f"{i}" for i in labels], yticklabels=[f"{i}" for i in labels],
                cbar=True, square=True)
    plt.title(f"Confusion Matrix ({mode})\nOA={oa*100:.2f}%, Kappa={kappa:.3f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, f"confusion_matrix_{safe_mode}.png"), dpi=300)
    plt.close()
    return {"mode": mode, "OA": oa, "Kappa": kappa, "MacroRecall": macro_recall}


def run_experiment(name, DATA, Y_train, Y_val, Y_test, output_root, cfg,
                    hsi_field=None, use_sen2=False, use_lst=False):
    print(f"\n==== Running {name}  (hsi={hsi_field}, sen2={use_sen2}, lst={use_lst}) ====")

    hsi = DATA[hsi_field] if hsi_field else None
    msi = DATA["sen2"] if use_sen2 else None
    lst_data = DATA.get("lst") if use_lst else None
    if use_lst and lst_data is None:
        raise SystemExit("Experiment uses LST, but no 'lst' dataset was found in all H5 files.")

    def make_ds(split_i, labels, augment=False):
        return PatchDataset(
            X1=hsi[split_i] if hsi is not None else None,
            X2=msi[split_i] if msi is not None else None,
            X3=lst_data[split_i] if lst_data is not None else None,
            labels=labels, augment=augment,
        )

    train_set = make_ds(0, Y_train, augment=True)
    val_set = make_ds(1, Y_val)
    test_set = make_ds(2, Y_test)

    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hsi_bands = hsi[0].shape[-1] if hsi is not None else 0
    msi_bands = msi[0].shape[-1] if msi is not None else 0
    lst_bands = lst_data[0].shape[-1] if lst_data is not None else 0
    spectral_size = train_set.X.shape[2]

    if hsi is not None and msi is not None:
        modal_type = "both"
    elif hsi is not None:
        modal_type = "hsi"
    elif msi is not None:
        modal_type = "msi"
    else:
        raise ValueError("Use at least HSI or Sentinel-2. LST-only is not configured for this model.")

    model = LCZ_HMSSNet(
        in_channels=1, spectral_size=spectral_size, num_classes=cfg.get("num_classes", 10),
        num_tokens=4, dim=64, modal_type=modal_type,
        hsi_bands=hsi_bands if hsi_bands else 1, msi_bands=msi_bands if msi_bands else 1,
        lst_bands=lst_bands, use_lst=use_lst,
    )

    es_path = os.path.join(output_root, f"best_model_{clean_name(name)}.pth")
    es = EarlyStopping(patience=cfg.get("patience", 10), path=es_path)
    train_model(model, train_loader, val_loader, max_epochs=cfg.get("max_epochs", 100), es=es)
    return evaluate_model(model, test_loader, output_root, cfg.get("num_classes", 10), mode=name)


def run_train(h5_dir: str, output_dir: str, cfg: dict) -> list[dict]:
    """Load H5 patch dataset(s), then run every experiment listed in cfg['experiments'].
    Returns the summary list (also written to <output_dir>/summary.csv)."""
    os.makedirs(output_dir, exist_ok=True)
    print("Device:", DEVICE)

    h5_files = collect_h5_files(h5_dir)
    sen2, hsi_bs, hsi_pca, lst, lbl, split_values = load_dataset(h5_files, cfg.get("sen2_scale", 10000.0))

    random_seed = cfg.get("random_seed", 42)
    predefined = split_indices_from_dataset(split_values)
    if predefined is None:
        idx = np.arange(len(lbl))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=random_seed, stratify=lbl)
        idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=random_seed,
                                               stratify=lbl[idx_train])
        print("Split: random stratified split")
    else:
        idx_train, idx_val, idx_test = predefined
        print("Split: using predefined spatial/group split from H5 dataset 'split'")

    def split3(arr):
        return arr[idx_train], arr[idx_val], arr[idx_test]

    DATA = {"sen2": split3(sen2), "hsi_bs": split3(hsi_bs)}
    if hsi_pca is not None:
        DATA["hsi_pca"] = split3(hsi_pca)
    if lst is not None:
        DATA["lst"] = split3(lst)   # already normalized in the H5; not standardized again here
        print("LST used as-is (already normalized in the H5)")

    Y_train, Y_val, Y_test = lbl[idx_train], lbl[idx_val], lbl[idx_test]
    print(f"Split sizes: train={len(Y_train)}  val={len(Y_val)}  test={len(Y_test)}")

    experiments = cfg["experiments"]
    summary = []
    for exp in experiments:
        summary.append(run_experiment(
            exp["name"], DATA, Y_train, Y_val, Y_test, output_dir, cfg,
            hsi_field=exp.get("hsi_field"), use_sen2=exp.get("use_sen2", False),
            use_lst=exp.get("use_lst", False),
        ))

    print("\n==== Summary ====")
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    by_mode = {r["mode"]: r for r in summary}
    for r in summary:
        base = r["mode"][: -len("+LST")] if r["mode"].endswith("+LST") else None
        if base and base in by_mode:
            b, l = by_mode[base], r
            print(f"\n[Delta LST] {r['mode']} vs {base}:  "
                  f"dOA={l['OA']-b['OA']:+.4f}  dKappa={l['Kappa']-b['Kappa']:+.4f}  "
                  f"dMacroRecall={l['MacroRecall']-b['MacroRecall']:+.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate LCZ_HMSSNet")
    parser.add_argument("--h5-dir", required=True, help="Single .h5 file or a directory of .h5 files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True, help="YAML with num_classes/batch_size/experiments/...")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_train(args.h5_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()
