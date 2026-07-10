# heatwise-lcz-classification

Stage 3 of the HEATWISE pipeline: the `LCZ_HMSSNet` model, its training/
evaluation loop, and sliding-window whole-scene inference. LST is a single
opt-in toggle everywhere instead of a parallel set of `_lst`-suffixed files.

## Install

```bash
pip install -r requirements.txt
```

## Entry point

`processor.py` (repo root) is the single command-line entry point (the one a
Docker container or CWL description would call), with `train` / `predict`
subcommands wrapping the internal package `src/heatwise_lcz_classification/`
below.

```bash
python processor.py train --h5-dir /path/to/Patch/Berlin_block500.h5 \
                           --output-dir results/Berlin \
                           --config config/train_config.example.yaml

python processor.py predict --config config/predict_config.example.yaml
```

`--h5-dir` accepts a single `.h5` (from `heatwise-patch-extraction`) or a
directory of several, which get concatenated. Each entry in the config's
`experiments` list runs one modality combination (`hsi_field` / `use_sen2` /
`use_lst`); `use_lst: true` only takes effect if the H5 actually has an `lst`
dataset. A predefined geographically-isolated `split` in the H5 is used
automatically; otherwise falls back to a random stratified split.

For `predict`, point `inputs.hsi` / `inputs.sen2` / `inputs.lst` at the same
rasters used for training (the LST raster is expected to already be resampled
onto the HSI's grid and normalized, i.e. the output of
`heatwise-hsi-lst-prep`'s `lst` step), set `weights` to a `.pth` produced by
`train`, and match `modal`/`use_lst`/`model.*` to how that model was trained.

`train.py` / `predict_map.py` also stay runnable directly (e.g. for
debugging) at their new location:

```bash
python src/heatwise_lcz_classification/train.py --h5-dir ... --output-dir ... --config ...
python src/heatwise_lcz_classification/predict_map.py --config ...
```

## Files

- `processor.py`: unified CLI entry point (`train` / `predict` subcommands).
- `src/heatwise_lcz_classification/model.py`: `LCZ_HMSSNet` (HSI / Sentinel-2 /
  optional LST branch), replaces the old `model.py` + `model_lst.py` pair —
  `use_lst` defaults to `False`.
- `src/heatwise_lcz_classification/train.py`: config-driven training/eval
  loop, replaces `train.py` + `train_lst.py`. Exposes
  `run_train(h5_dir, output_dir, cfg)`, called by `processor.py train`.
- `src/heatwise_lcz_classification/predict_map.py`: sliding-window
  whole-scene inference, replaces `lcz_map.py` + `lcz_map_helsinki_lst.py`
  (no more Helsinki-specific hardcoding). Exposes `run_predict(cfg)`, called
  by `processor.py predict`.

## Sample data

`data/Berlin/` holds a small (~20 MB) self-contained bundle for quick local
testing: `Berlin_sample.h5` (42 patches, from `heatwise-patch-extraction`'s
own sample data), the matching `Berlin_hsi_bs.tif`/`Berlin_S2_..._sample.tif`
rasters (for `predict`), and `best_model_HSI-BS_sample.pth` -- a checkpoint
trained for 2 epochs on that same tiny H5 (not scientifically useful, just
enough to exercise `predict` without retraining). `examples/train_config_sample.yaml`
and `examples/predict_config_sample.yaml` are matching configs. Run from the
repo root:

```bash
python processor.py train --h5-dir data/Berlin/Berlin_sample.h5 \
  --output-dir examples/output/train --config examples/train_config_sample.yaml

python processor.py predict --config examples/predict_config_sample.yaml
```

## Docker

The image is built under its release-shaped name (registry namespace +
versioned tag, matching both CWL files' `dockerPull`), so local tests
exercise the exact tag that will later be pushed to the registry:

```bash
docker build -t ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0 .

docker run --rm -v /path/to/host/output:/app/output \
  ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0 \
  train --h5-dir data/Berlin/Berlin_sample.h5 \
        --output-dir /app/output/train --config examples/train_config_sample.yaml

docker run --rm -v /path/to/host/output:/app/output \
  ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0 \
  predict --config examples/predict_config_sample.yaml --output /app/output/Berlin_LCZ.tif
```

Base image: `python:3.11-slim`; `torch` is installed from the CPU wheel index
to keep the image size reasonable (EOAP platform execution isn't assumed to
have GPU access -- swap in the CUDA wheel index yourself if you need GPU
training/inference in Docker).

> The image has since been built and exercised repeatedly through `cwltool`
> runs (both standalone and as the `train`/`predict` steps of
> `heatwise-lcz-pipeline`).

## CWL

This repo has **two** CWL files (unlike the other two HEATWISE repos, which
each have one) because `train` and `predict` have very different inputs and
outputs: `heatwise_lcz_train.cwl` and `heatwise_lcz_predict.cwl`, both
wrapping the same `processor.py` entry point. `examples/job_train.yaml` /
`examples/job_predict.yaml` are ready-to-use job orders for the bundled
sample data:

```bash
cd examples
cwltool ../heatwise_lcz_train.cwl job_train.yaml
cwltool ../heatwise_lcz_predict.cwl job_predict.yaml
```

Like `heatwise-patch-extraction`, the `config` file's `inputs.*`/`weights`
paths resolve against the container's working directory, which under
`cwltool` is an empty per-job staging directory, NOT the image's
`WORKDIR /app` (confirmed with `heatwise-hsi-lst-prep`'s CWL: a bare
relative `processor.py` failed until pointed at it via an absolute
`/app/processor.py`). So:

- Both CWL files' `arguments` reference `/app/processor.py` by absolute path.
- `train_config_sample.yaml` has no file paths inside it (`h5_dir` is passed
  as an explicit CWL File/Directory input, `output_dir` is relative to
  `cwltool`'s own workdir -- both unaffected), so `heatwise_lcz_train.cwl`
  uses it as-is, no `_docker` variant needed.
- `predict_config_sample.yaml` *does* have relative raster/checkpoint paths
  (`inputs.hsi`, `inputs.sen2`, `weights`), so there's a
  `predict_config_sample_docker.yaml` with absolute `/app/data/...`
  equivalents; `examples/job_predict.yaml` is wired to it.
  `predict_config_sample.yaml` (relative paths) is for local/non-Docker
  runs only.

> **Rebuild the image before testing this** (`docker build -t
> ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0 .`): the Dockerfile's `ENTRYPOINT` was
> just fixed too (`processor.py` -> `/app/processor.py`, absolute), for the
> same reason described in `heatwise-patch-extraction`'s README (an actual
> `cwltool` run against that repo's identical setup failed until fixed).
> Both CWL files here had their redundant `baseCommand`/`python
> /app/processor.py` arguments removed accordingly -- they now only
> contribute the `train`/`predict` subcommand + flags, since ENTRYPOINT
> already supplies `python /app/processor.py`.
