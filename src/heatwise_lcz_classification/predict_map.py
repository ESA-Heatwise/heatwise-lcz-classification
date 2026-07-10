#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run sliding-window inference over a whole scene with a trained model to
produce an LCZ classification map (GeoTIFF). Merged from the old lcz_map.py
(no LST) and lcz_map_helsinki_lst.py (LST, Helsinki-specific), with the
city-specific hardcoding removed and LST turned into a generic toggle.

Preprocessing must match train.py exactly:
  - Sentinel-2 / sen2_scale; hsi kept at its raw values; LST kept at its raw
    values (already reprojected onto the same 10m grid as HSI and normalized
    by heatwise-hsi-lst-prep, so it is read directly here without being
    normalized again);
  - channel order: [HSI, Sen2(, LST)] (matches the split inside model.py);
  - fixed patch size, center pixel = the patch's predicted class;
  - class_order must match training, to map model output indices back to LCZ codes.

Usage:
    python predict_map.py --config config/predict_config.example.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import rasterio
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from model import LCZ_HMSSNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LCZ_COLORS = {
    1: (140, 0, 0), 2: (209, 0, 0), 3: (255, 0, 0),
    4: (191, 77, 0), 5: (255, 102, 0), 6: (255, 153, 85),
    7: (250, 238, 5), 8: (188, 188, 188), 9: (255, 204, 170),
    10: (85, 85, 85),
    11: (0, 106, 0), 12: (0, 170, 0), 13: (100, 133, 37),
    14: (185, 219, 121), 15: (0, 0, 0), 16: (251, 247, 174),
    17: (106, 106, 255),
}
LCZ_NAMES = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
             11: "A", 12: "B", 13: "C", 14: "D", 15: "E", 16: "F", 17: "G"}


def read_hsi(path):
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        prof = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0
    return arr, prof, nodata


def fill_interior_holes(hsi, nodata):
    valid = np.all(hsi != nodata, axis=0) & np.all(np.isfinite(hsi), axis=0)
    interior = binary_fill_holes(valid) & ~valid
    if not interior.any():
        return hsi, 0
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    out = hsi.copy()
    for c in range(hsi.shape[0]):
        out[c][interior] = hsi[c][iy[interior], ix[interior]]
    return out, int(interior.sum())


def read_on_grid(path, ref_prof, band=None, resampling=Resampling.nearest, dst_nodata=0):
    """Resample any raster (single- or multi-band) onto the reference grid, returning (C,H,W)."""
    H, W = ref_prof["height"], ref_prof["width"]
    with rasterio.open(path) as src:
        bands = [band] if band is not None else list(range(1, src.count + 1))
        out = np.full((len(bands), H, W), dst_nodata, dtype="float32")
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(src, b), destination=out[i],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
                resampling=resampling, src_nodata=src.nodata, dst_nodata=dst_nodata,
            )
    return out


def valid_patch_mask(valid_pix, patch):
    """For each top-left corner (tl), check whether the patch x patch window is
    fully valid (vectorized via an integral/summed-area image)."""
    invalid = (~valid_pix).astype(np.int64)
    H, W = invalid.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.int64)
    ii[1:, 1:] = invalid.cumsum(0).cumsum(1)
    p = patch
    A = ii[p:H + 1, p:W + 1]
    B = ii[0:H - p + 1, p:W + 1]
    C = ii[p:H + 1, 0:W - p + 1]
    D = ii[0:H - p + 1, 0:W - p + 1]
    return (A - B - C + D) == 0


def build_model(modal, hsi_bands, msi_bands, lst_bands, use_lst, num_classes):
    if modal == "both":
        hb, mb = hsi_bands, msi_bands
        spectral = hb + mb
    elif modal == "hsi":
        hb, mb = hsi_bands, 1
        spectral = hb
    else:  # 'msi'
        hb, mb = 1, msi_bands
        spectral = mb
    if use_lst:
        spectral += lst_bands
    return LCZ_HMSSNet(
        in_channels=1, spectral_size=spectral, num_classes=num_classes,
        num_tokens=4, dim=64, modal_type=modal,
        hsi_bands=hb, msi_bands=mb, lst_bands=lst_bands if use_lst else 0, use_lst=use_lst,
    )


def save_png_preview(out, png_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    codes = sorted(LCZ_COLORS.keys())
    code_to_idx = {c: i for i, c in enumerate(codes)}
    idx_img = np.full(out.shape, -1, dtype=int)
    for c, i in code_to_idx.items():
        idx_img[out == c] = i
    cmap = ListedColormap([tuple(v / 255 for v in LCZ_COLORS[c]) for c in codes])
    cmap.set_under((1, 1, 1, 0))
    norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.where(idx_img < 0, np.nan, idx_img), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    present = [c for c in codes if np.any(out == c)]
    handles = [Patch(facecolor=tuple(v / 255 for v in LCZ_COLORS[c]), edgecolor="k", label=LCZ_NAMES[c])
               for c in present]
    ax.legend(handles=handles, title="LCZ", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[predict_map] Saved color preview:", png_path)


def run_predict(cfg: dict) -> None:
    modal = cfg.get("modal", "both")
    use_lst = bool(cfg.get("use_lst", False))
    inputs = cfg["inputs"]
    model_cfg = cfg.get("model", {})
    class_order = [int(c) for c in cfg["class_order"]]
    sen2_scale = cfg.get("sen2_scale", 10000.0)
    patch_size = cfg.get("patch_size", 32)
    stride = cfg.get("stride", 4)
    batch_size = cfg.get("batch_size", 256)
    fill_holes = cfg.get("fill_interior_holes", False)
    map_nodata = cfg.get("map_nodata", 0)
    save_png = cfg.get("save_png", True)
    output_path = cfg["output"]

    print(f"[predict_map] modal={modal}  use_lst={use_lst}  device={DEVICE}")

    half = patch_size // 2
    hsi, prof, hsi_nodata = read_hsi(inputs["hsi"])
    H, W = prof["height"], prof["width"]
    n_filled = 0
    if fill_holes:
        hsi, n_filled = fill_interior_holes(hsi, hsi_nodata)
        if n_filled:
            print(f"[predict_map] Nearest-neighbor filled interior holes: {n_filled} pixels")
    hsi_valid = np.all(hsi != hsi_nodata, axis=0) & np.all(np.isfinite(hsi), axis=0)

    layers = []
    valid = hsi_valid
    if modal in ("both", "hsi"):
        layers.append(np.transpose(hsi, (1, 2, 0)))
    if modal in ("both", "msi"):
        sen2 = read_on_grid(inputs["sen2"], prof, resampling=Resampling.nearest, dst_nodata=0) / sen2_scale
        layers.append(np.transpose(sen2, (1, 2, 0)))

    lst_bands = model_cfg.get("lst_bands", 1)
    if use_lst:
        # The LST raster has already been resampled + normalized onto the same
        # 10m grid as the HSI by heatwise-hsi-lst-prep.
        lst_arr = read_on_grid(inputs["lst"], prof, band=1, resampling=Resampling.nearest, dst_nodata=-9999.0)
        lst_ok = (lst_arr[0] != -9999.0) & np.isfinite(lst_arr[0])
        layers.append(np.transpose(lst_arr, (1, 2, 0)))
        valid = valid & lst_ok
        print(f"[predict_map] LST valid pixel fraction: {100 * lst_ok.mean():.1f}%")

    stack = np.concatenate(layers, axis=2).astype("float32")
    C_total = stack.shape[2]
    print(f"[predict_map] Image {H}x{W}, stacked channels={C_total}, valid pixels={int(valid.sum())}")

    ok = valid_patch_mask(valid, patch_size)
    if stride > 1:
        step = np.zeros_like(ok)
        step[::stride, ::stride] = True
        ok = ok & step
    tls = np.argwhere(ok)
    print(f"[predict_map] Windows to predict: {len(tls)}")
    if len(tls) == 0:
        raise SystemExit("No valid windows to predict; check the image/nodata.")

    swv = sliding_window_view(stack, (patch_size, patch_size), axis=(0, 1))
    out = np.full((H, W), map_nodata, dtype="uint8")
    code_lut = np.array(class_order, dtype="uint8")

    model = build_model(
        modal=modal, hsi_bands=model_cfg["hsi_bands"], msi_bands=model_cfg["msi_bands"],
        lst_bands=lst_bands, use_lst=use_lst, num_classes=model_cfg["num_classes"],
    )
    state = torch.load(cfg["weights"], map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(
            f"[predict_map] Failed to load weights {cfg['weights']!r} into a modal={modal!r} "
            f"use_lst={use_lst} model. This almost always means `modal`/`use_lst`/`model.*` in "
            f"the config don't match how these weights were trained (e.g. the weights file name "
            f"suggests LST but use_lst is false here, or vice versa). Original error:\n{e}"
        )
    model.to(DEVICE).eval()

    with torch.no_grad():
        for s in range(0, len(tls), batch_size):
            chunk = tls[s:s + batch_size]
            patches = swv[chunk[:, 0], chunk[:, 1]]
            xb = torch.from_numpy(np.ascontiguousarray(patches)).float().unsqueeze(1).to(DEVICE)
            pred_idx = model(xb).argmax(1).cpu().numpy()
            codes = code_lut[pred_idx]
            cr = chunk[:, 0] + half
            cc = chunk[:, 1] + half
            if stride == 1:
                out[cr, cc] = codes
            else:
                o = stride // 2
                for r, c, v in zip(cr, cc, codes):
                    out[max(r - o, 0):r - o + stride, max(c - o, 0):c - o + stride] = v
            if (s // batch_size) % 50 == 0:
                print(f"  progress {min(s + batch_size, len(tls))}/{len(tls)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    prof.update(count=1, dtype="uint8", nodata=map_nodata,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    colormap = {code: (r, g, b, 255) for code, (r, g, b) in LCZ_COLORS.items()}
    colormap[map_nodata] = (0, 0, 0, 0)
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.write_colormap(1, colormap)
        dst.update_tags(class_order=",".join(map(str, class_order)), modal=modal, use_lst=str(use_lst))
    uniq, cnt = np.unique(out[out != map_nodata], return_counts=True)
    print("\n[predict_map] Saved LCZ map:", output_path)
    print("Per-class pixel counts:", dict(zip(uniq.tolist(), cnt.tolist())))

    if save_png:
        title = cfg.get("title", os.path.basename(output_path))
        save_png_preview(out, os.path.splitext(output_path)[0] + "_preview.png", title)


def main():
    parser = argparse.ArgumentParser(description="Sliding-window LCZ map inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_predict(cfg)


if __name__ == "__main__":
    main()
