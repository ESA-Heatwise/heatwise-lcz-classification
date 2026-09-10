# heatwise-lcz-classification

Stage 3 of the HEATWISE LCZ pipeline: LCZ_HMSSNet training/evaluation and
sliding-window whole-scene inference.

The repository exposes two operations through the same `processor.py` entry
point:

- `train`: trains and evaluates LCZ_HMSSNet using geographically isolated
  patch datasets produced by `heatwise-patch-extraction`;
- `predict`: performs whole-scene LCZ inference using aligned EO raster
  products and a trained model checkpoint.

Both operations are exposed as EOAP-compatible CWL Application Packages with
STAC staging and stage-out.

## Install

```bash
pip install -r requirements.txt
```

## Entry point

`processor.py` is the common command-line entry point and provides two
subcommands:

```text
train
predict
```

### Local training

For local/non-EOAP execution, training can still use a patch HDF5 file or a
directory containing several HDF5 datasets:

```bash
python processor.py train \
  --h5-dir data/Berlin/Berlin_patches.h5 \
  --output-dir examples/output/train \
  --config examples/train_config_sample.yaml
```

`--h5-dir` accepts either a single `.h5` file or a directory of HDF5 files.

Each entry in the configuration's `experiments` list defines one modality
combination through parameters such as:

- `hsi_field`
- `use_sen2`
- `use_lst`

If the HDF5 dataset contains a predefined geographically isolated `split`,
that split is used for training, validation, and testing.

### Local prediction

The original configuration-based prediction interface also remains available:

```bash
python processor.py predict \
  --config examples/predict_config_sample.yaml
```

For local execution, the configuration can contain the EO raster paths under
`inputs` together with the model checkpoint path under `weights`.

`train.py` and `predict_map.py` also remain independently runnable for
development and debugging.

## Repository structure

Important components include:

- `processor.py`: unified CLI entry point for `train` and `predict`;
- `src/heatwise_lcz_classification/model.py`: LCZ_HMSSNet model;
- `src/heatwise_lcz_classification/train.py`: training and evaluation logic;
- `src/heatwise_lcz_classification/predict_map.py`: sliding-window
  whole-scene inference;
- `src/heatwise_lcz_classification/stac_io.py`: STAC staging and stage-out
  helpers;
- `heatwise_lcz_train.cwl`: EOAP Application Package for training;
- `heatwise_lcz_predict.cwl`: EOAP Application Package for prediction.

## Sample data

`data/Berlin/` contains a self-contained sample bundle for exercising both
training and prediction:

```text
Berlin_S2.tif
Berlin_hsi_bs.tif
Berlin_lst_final.tif
Berlin_patches.h5
best_model_HSI-BS.pth

catalog.json
Berlin_patches_item.json
Berlin_inputs_item.json
```

The two STAC Items serve different purposes:

- `Berlin_patches_item.json` exposes the `patch_h5` asset used for training;
- `Berlin_inputs_item.json` exposes the `hsi`, `sentinel2`, and optional `lst`
  EO assets used for prediction.

Both are linked from the same `catalog.json`.

The trained model checkpoint `best_model_HSI-BS.pth` is intentionally kept
separate from the EO raster STAC assets and is supplied explicitly to the
prediction workflow as a CWL `File`.

## EOAP training input

For EOAP execution, training receives a staged STAC `Directory` containing a
`catalog.json`.

The relevant STAC Item must expose:

```text
patch_h5
```

For example:

```text
input_catalog/
├── catalog.json
├── Berlin_patches_item.json
└── Berlin_patches.h5
```

The processor receives:

```text
--input-catalog <staged-directory>/catalog.json
```

and resolves the `patch_h5` asset before calling the existing training logic.

The training configuration remains separate and contains only training
parameters and experiment definitions.

## Training output

Training produces artifacts such as:

```text
best_model_<experiment>.pth
summary.csv
confusion matrices
per-class accuracy files
other evaluation outputs
```

After training, `processor.py` generates a STAC catalog describing the
training artifacts:

```text
catalog.json
training_artifacts_item.json
```

The training artifacts and STAC metadata are written together for EOAP
stage-out.

## EOAP prediction input

Prediction receives three distinct inputs:

```text
config         → inference parameters
input_catalog  → staged EO raster products
weights        → trained model checkpoint
```

The STAC Item used for prediction must contain:

- `hsi`: required hyperspectral raster;
- `sentinel2`: required Sentinel-2 raster;
- `lst`: optional LST raster.

For example:

```text
input_catalog/
├── catalog.json
├── Berlin_inputs_item.json
├── Berlin_hsi_bs.tif
├── Berlin_S2.tif
└── Berlin_lst_final.tif
```

The model checkpoint is supplied separately:

```text
best_model_HSI-BS.pth
```

During EOAP execution, `processor.py` resolves the STAC assets into the
configuration's internal `inputs` structure and overrides `weights` with the
explicit checkpoint supplied by CWL.

## Prediction output

Whole-scene prediction generates an LCZ classification GeoTIFF:

```text
Berlin_LCZ_sample.tif
```

and, when `save_png: true`, an optional colour preview:

```text
Berlin_LCZ_sample_preview.png
```

The processor also writes:

```text
lcz_prediction_item.json
catalog.json
```

The generated STAC Item describes the LCZ GeoTIFF and optional preview.

## Configuration

### Training

The bundled training configuration is:

```text
examples/train_config_sample.yaml
```

It contains model-training parameters such as:

- number of classes;
- batch size;
- maximum epochs;
- early stopping;
- Sentinel-2 scaling;
- random seed;
- modality experiments.

The training HDF5 path is not stored in the configuration for EOAP execution;
it is resolved from the staged STAC catalog.

### Prediction

For local execution:

```text
examples/predict_config_sample.yaml
```

can contain raster and checkpoint paths directly.

For Docker/CWL execution:

```text
examples/predict_config_sample_docker.yaml
```

contains only prediction parameters. EO raster paths and model weights are
intentionally excluded because they are supplied separately through the CWL
inputs.

## Docker

Build the versioned processor image with:

```bash
docker build \
  -t ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.1 \
  .
```

The image uses `CMD` as its default command. CWL therefore controls the actual
processor command explicitly.

### Docker training example

```bash
mkdir -p output/train

docker run --rm \
  -v "$(pwd)/data/Berlin:/input:ro" \
  -v "$(pwd)/output/train:/output" \
  ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.1 \
  python /app/processor.py train \
  --input-catalog /input/catalog.json \
  --output-dir /output \
  --config /app/examples/train_config_sample.yaml
```

### Docker prediction example

```bash
mkdir -p output/predict

docker run --rm \
  -v "$(pwd)/data/Berlin:/input:ro" \
  -v "$(pwd)/output/predict:/output" \
  ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.1 \
  python /app/processor.py predict \
  --config /app/examples/predict_config_sample_docker.yaml \
  --input-catalog /input/catalog.json \
  --weights /input/best_model_HSI-BS.pth \
  --output /output/Berlin_LCZ_sample.tif
```

The image is based on `python:3.11-slim`. PyTorch is installed from the CPU
wheel index so the EOAP processor does not require GPU availability.

## EO Application Packages / CWL

This repository provides two CWL v1.2 Application Packages because training
and prediction have different inputs and execution semantics:

```text
heatwise_lcz_train.cwl
heatwise_lcz_predict.cwl
```

Each package contains:

- a top-level `Workflow` with `id: main`;
- a dedicated `CommandLineTool`;
- explicit `baseCommand` and processor arguments;
- a versioned `DockerRequirement`;
- documented workflow inputs;
- STAC-based staging for EO-derived products;
- a single `Directory` stage-out using `glob: "."`.

### Training Application Package

The training workflow inputs are:

```text
input_catalog  Directory
config         File
output_dir     string
```

Its processor command is:

```text
python /app/processor.py train
```

The staged catalog path is passed as:

```text
--input-catalog <staged-directory>/catalog.json
```

The complete CWL working directory is returned as:

```yaml
outputs:
  output:
    type: Directory
    outputBinding:
      glob: "."
```

### Prediction Application Package

The prediction workflow inputs are:

```text
config         File
input_catalog  Directory
weights        File
output         string
```

Its processor command is:

```text
python /app/processor.py predict
```

The EO raster products are resolved from:

```text
<input_catalog>/catalog.json
```

while the trained model is supplied separately through the `weights` input.

The complete working directory is returned using:

```yaml
outputs:
  output:
    type: Directory
    outputBinding:
      glob: "."
```

## CWL examples

The bundled CWL jobs are:

```text
examples/job_train.yaml
examples/job_predict.yaml
```

### Training

```yaml
input_catalog:
  class: Directory
  path: ../data/Berlin

config:
  class: File
  path: train_config_sample.yaml

output_dir: "."
```

Run with:

```bash
cd examples
cwltool ../heatwise_lcz_train.cwl job_train.yaml
```

### Prediction

```yaml
config:
  class: File
  path: predict_config_sample_docker.yaml

input_catalog:
  class: Directory
  path: ../data/Berlin

weights:
  class: File
  path: ../data/Berlin/best_model_HSI-BS.pth

output: Berlin_LCZ_sample.tif
```

Run with:

```bash
cd examples
cwltool ../heatwise_lcz_predict.cwl job_predict.yaml
```

## Automated validation

The `eoap-compliance` branch includes a GitHub Actions workflow that validates
both Application Packages end-to-end.

The automated workflow performs:

- Python syntax validation;
- training CWL validation with `cwltool`;
- prediction CWL validation with `cwltool`;
- input STAC validation with PySTAC;
- Docker image build;
- end-to-end LCZ training;
- training-output STAC validation;
- verification of trained model checkpoints and `summary.csv`;
- end-to-end whole-scene LCZ prediction;
- prediction-output STAC validation;
- verification of the generated LCZ GeoTIFF and optional preview.

The bundled Berlin training and prediction examples have been successfully
executed through this complete validation workflow.

## Relationship to the HEATWISE LCZ pipeline

This repository operates after patch extraction:

```text
heatwise-hsi-lst-prep
        ↓
heatwise-patch-extraction
        ↓
heatwise-lcz-classification
        ├── train
        └── predict
```

For training, the STAC output of `heatwise-patch-extraction` can be staged as
the input catalog and its `patch_h5` asset is resolved automatically.

For prediction, aligned HSI/Sentinel-2/LST products are staged through STAC,
while the trained LCZ_HMSSNet checkpoint is supplied separately.
