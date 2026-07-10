cwlVersion: v1.2
class: CommandLineTool

label: HEATWISE LCZ Prediction
doc: >
  EOAP-compatible HEATWISE LCZ_HMSSNet whole-scene inference processor. Runs
  sliding-window inference over a full HSI (+Sentinel-2, +optional LST) scene
  with a trained checkpoint, producing an LCZ classification map GeoTIFF
  (+ optional colour preview PNG).

  As with heatwise-hsi-lst-prep's config, the `config` file's `inputs.*`
  and `weights` paths are resolved against the container's working
  directory. Under `cwltool` that's an empty per-job staging directory, NOT
  the image's Dockerfile `WORKDIR /app` (confirmed with heatwise-hsi-lst-prep's
  CWL). So for CWL use, every path *inside* the config must be an absolute
  `/app/...` path into the image (baked in via `COPY . .`) -- see
  `examples/predict_config_sample_docker.yaml`.
  `examples/predict_config_sample.yaml` (relative `data/...` paths) is for
  local/non-Docker runs only.

requirements:
  DockerRequirement:
    # Release-shaped image reference. Before publishing, build/tag this
    # image locally with the same name so local cwltool runs exercise the
    # exact tag that will later be pushed to the registry.
    dockerImageId: ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0
    dockerPull: ghcr.io/heatwise-lcz/heatwise-lcz-classification:0.1.0

arguments:
  # No baseCommand/`python /app/processor.py` here on purpose -- see
  # heatwise_lcz_train.cwl's note: the image's ENTRYPOINT already supplies
  # `python /app/processor.py`, and `docker run` arguments are appended to
  # ENTRYPOINT rather than replacing it, so this CWL only contributes the
  # `predict` subcommand + flags.
  - predict

inputs:
  config:
    type: File
    inputBinding:
      prefix: --config
    doc: YAML with modal/use_lst/inputs/model/weights/class_order/... (see examples/predict_config_sample_docker.yaml for the CWL/Docker variant).

  output:
    type: string
    default: output/lcz_map.tif
    inputBinding:
      prefix: --output
    doc: Output LCZ map GeoTIFF path (relative to the container working directory), overrides the config's `output`.

outputs:
  lcz_map:
    type: File
    outputBinding:
      glob: $(inputs.output)
    doc: LCZ classification map (GeoTIFF, colormap embedded).

  lcz_map_preview:
    type: File?
    outputBinding:
      glob: "**/*_preview.png"
    doc: Optional colour preview PNG (only produced if the config sets save_png true).
