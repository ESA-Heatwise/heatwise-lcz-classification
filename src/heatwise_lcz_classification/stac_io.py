#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HEATWISE LCZ classification processor -- single command-line entry point.

Wraps train.py (train/evaluate LCZ_HMSSNet on patch H5 datasets) and
predict_map.py (sliding-window whole-scene inference) behind two subcommands,
so this repo exposes the same "one processor.py entry point" shape as
heatwise-hsi-lst-prep and heatwise-patch-extraction (and matches the EOAP
reference structure: a single processor.py callable by a user, Docker
container, or CWL description).

Subcommands:
  train      Train/evaluate LCZ_HMSSNet on one or more patch H5 files.
  predict    Sliding-window whole-scene inference -> LCZ map GeoTIFF.

train.py and predict_map.py remain runnable standalone (unchanged CLI), for
anyone already using them directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent
_PKG_DIR = _REPO_ROOT / "src" / "heatwise_lcz_classification"
sys.path.insert(0, str(_REPO_ROOT))
# train.py/predict_map.py import their sibling model.py with a plain
# `from model import ...` (so they also stay independently runnable as
# scripts); add their package directory too so that import resolves the
# same way when processor.py imports them as package members.
sys.path.insert(0, str(_PKG_DIR))

from src.heatwise_lcz_classification.train import run_train
from src.heatwise_lcz_classification.predict_map import run_predict
from src.heatwise_lcz_classification.stac_io import (
    patch_h5_from_stac,
    prediction_inputs_from_stac,
    write_training_catalog,
    write_prediction_catalog,
)


def cmd_train(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.input_catalog:
        h5_input = patch_h5_from_stac(args.input_catalog)
    else:
        h5_input = args.h5_dir

    run_train(h5_input, args.output_dir, cfg)

    catalog_path = write_training_catalog(args.output_dir)

    print(f"[processor] Training input: {h5_input}")
    print(f"[processor] Output STAC catalog: {catalog_path}")


def cmd_predict(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.input_catalog:
        cfg["inputs"] = prediction_inputs_from_stac(
            args.input_catalog
        )

    if args.weights:
        cfg["weights"] = args.weights

    if args.output:
        cfg["output"] = args.output

    run_predict(cfg)

    catalog_path = write_prediction_catalog(cfg["output"])

    print(f"[processor] Output LCZ map: {cfg['output']}")
    print(f"[processor] Output STAC catalog: {catalog_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEATWISE LCZ classification processor")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "train",
        help="Train/evaluate LCZ_HMSSNet on patch H5 file(s).",
    )
    
    train_input = p.add_mutually_exclusive_group(required=True)
    
    train_input.add_argument(
        "--h5-dir",
        help=(
            "Single .h5 file or directory of .h5 files. "
            "Used for local/non-EOAP execution."
        ),
    )
    
    train_input.add_argument(
        "--input-catalog",
        help=(
            "Path to the staged STAC catalog.json containing the "
            "`patch_h5` training asset."
        ),
    )
    
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints, metrics, and training artifacts.",
    )
    
    p.add_argument(
        "--config",
        required=True,
        help="YAML with num_classes/batch_size/experiments/...",
    )
    
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "predict",
        help="Sliding-window whole-scene inference -> LCZ map GeoTIFF.",
    )
    
    p.add_argument(
        "--config",
        required=True,
        help="YAML with modal/use_lst/model/class_order/... parameters.",
    )
    
    p.add_argument(
        "--input-catalog",
        help=(
            "Path to the staged STAC catalog.json containing the HSI, "
            "Sentinel-2, and optional LST input assets."
        ),
    )
    
    p.add_argument(
        "--weights",
        help="Trained model checkpoint (.pth). Overrides config `weights`.",
    )
    
    p.add_argument(
        "--output",
        help=(
            "Overrides the config's `output` LCZ GeoTIFF path if given."
        ),
    )
    
    p.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
