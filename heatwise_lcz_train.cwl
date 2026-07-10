cwlVersion: v1.2
class: CommandLineTool

label: HEATWISE LCZ Training
doc: >
  EOAP-compatible HEATWISE LCZ_HMSSNet training/evaluation processor. Trains
  and evaluates one or more modality-combination experiments (defined in the
  config's `experiments:` list) on a patch H5 dataset, writing checkpoints,
  confusion matrices and a summary.csv per experiment.

  Note: this repo exposes two operations (train / predict) with very
  different inputs and outputs, so -- unlike heatwise-hsi-lst-prep and
  heatwise-patch-extraction, which each have one .cwl -- there are two CWL
  files here: this one and heatwise_lcz_predict.cwl. Both wrap the same
  processor.py entry point (`train` / `predict` subcommands).

requirements:
  DockerRequirement:
    # Release-shaped image reference. Before publishing, build/tag this
    # image locally with the same name so local cwltool runs exercise the
    # exact tag that will later be pushed to the registry.
    dockerImageId: ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0
    dockerPull: ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0

arguments:
  # No baseCommand/`python /app/processor.py` here on purpose: the image's
  # ENTRYPOINT is `["python", "/app/processor.py"]` (absolute path, so it
  # resolves regardless of cwltool overriding the working directory).
  # `docker run` arguments are *appended to* ENTRYPOINT, not a replacement
  # for it (unlike CMD) -- confirmed by an actual cwltool run against
  # heatwise-patch-extraction that failed until the redundant
  # `python`/`processor.py` were removed from the CWL side. So this CWL only
  # contributes the `train` subcommand + flags.
  #
  # train_config_sample.yaml itself has no file paths inside it (only
  # h5_dir, passed as an explicit CWL File/Directory input, and output_dir,
  # relative to cwltool's own workdir -- both unaffected by the workdir
  # issue), so no "_docker" config variant is needed here, unlike
  # heatwise_lcz_predict.cwl.
  - train

inputs:
  h5_dir:
    type: [File, Directory]
    inputBinding:
      prefix: --h5-dir
    doc: A single patch .h5 file (from heatwise-patch-extraction) or a directory of several.

  config:
    type: File
    inputBinding:
      prefix: --config
    doc: YAML with num_classes/batch_size/max_epochs/experiments/...

  output_dir:
    type: string
    default: output
    inputBinding:
      prefix: --output-dir

outputs:
  output_directory:
    type: Directory
    outputBinding:
      glob: $(inputs.output_dir)
    doc: Checkpoints (best_model_<experiment>.pth), confusion matrices, per-class accuracy CSVs, summary.csv.

  summary_csv:
    type: File
    outputBinding:
      glob: $(inputs.output_dir)/summary.csv
