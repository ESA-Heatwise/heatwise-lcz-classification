cwlVersion: v1.2

$namespaces:
  s: https://schema.org/

s:softwareVersion: 0.1.1
s:version: 0.1.1

schemas:
  - http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:

  # -------------------------------------------------------------------------
  # Main EOAP Workflow
  # -------------------------------------------------------------------------
  - class: Workflow
    id: main
    label: HEATWISE LCZ Training Workflow
    doc: |
      EOAP-compatible HEATWISE LCZ classification training workflow.

      The workflow trains and evaluates LCZ_HMSSNet experiments using a
      geographically isolated HDF5 patch dataset produced by the HEATWISE
      patch-extraction processor.

      The training dataset is provided through a staged STAC catalog
      directory. The processor generates trained model checkpoints,
      evaluation metrics, confusion matrices and summary files, together
      with an output STAC catalog describing the generated artifacts.

    requirements: []

    inputs:

      - id: input_catalog
        type: Directory
        label: input STAC catalog
        doc: |
          Directory containing a STAC catalog named catalog.json referencing
          the HDF5 patch dataset used for LCZ model training.

          The referenced STAC Item must expose the patch dataset through the
          `patch_h5` asset.

      - id: config
        type: File
        label: training configuration
        doc: |
          YAML configuration defining the LCZ training parameters, including
          number of classes, batch size, maximum epochs, early stopping,
          random seed, and modality experiments.

      - id: output_dir
        type: string
        label: output directory
        default: "."
        doc: |
          Output directory path relative to the CWL working directory.
          By default, training artifacts are written directly to the working
          directory for EOAP stage-out.

    steps:

      processor:
        run: "#lcz_train_processor"

        in:
          input_catalog: input_catalog
          config: config
          output_dir: output_dir

        out:
          - output

    outputs:

      output:
        type: Directory
        outputSource: processor/output


  # -------------------------------------------------------------------------
  # LCZ training CommandLineTool
  # -------------------------------------------------------------------------
  - class: CommandLineTool
    id: lcz_train_processor
    label: HEATWISE LCZ Training Processor
    doc: |
      HEATWISE LCZ_HMSSNet training and evaluation processor.

      The processor resolves the patch HDF5 dataset from the staged STAC
      catalog, trains the configured LCZ experiments, evaluates the resulting
      models, and writes the generated training artifacts together with an
      output STAC catalog.

    requirements:

      DockerRequirement:
        dockerPull: ghcr.io/esa-heatwise/heatwise-lcz-classification:eoap-compliance

      InlineJavascriptRequirement: {}

    baseCommand: python

    arguments:
      - /app/processor.py
      - train

    inputs:

      input_catalog:
        type: Directory
        label: input STAC catalog
        doc: |
          Directory containing catalog.json and the associated STAC Item
          referencing the HDF5 training dataset through a `patch_h5` asset.

          The path to catalog.json inside this directory is passed to the
          processor through the --input-catalog argument.
        inputBinding:
          prefix: --input-catalog
          valueFrom: $(self.path + "/catalog.json")

      config:
        type: File
        label: training configuration
        doc: |
          YAML configuration controlling LCZ_HMSSNet training and evaluation,
          including experiments, number of classes, batch size, epochs,
          early stopping, scaling, and random seed.
        inputBinding:
          prefix: --config

      output_dir:
        type: string
        label: output directory
        default: "."
        doc: |
          Output directory path relative to the CWL working directory.
        inputBinding:
          prefix: --output-dir

    outputs:

      output:
        type: Directory
        doc: |
          Complete CWL working directory containing the generated training
          artifacts, including model checkpoints, evaluation metrics,
          confusion matrices, summary files, and the generated STAC catalog.
        outputBinding:
          glob: "."
