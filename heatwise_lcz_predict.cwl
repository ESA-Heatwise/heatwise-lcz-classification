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
    label: HEATWISE LCZ Prediction Workflow
    doc: |
      EOAP-compatible HEATWISE LCZ whole-scene prediction workflow.

      The workflow performs sliding-window LCZ inference using aligned
      hyperspectral and Sentinel-2 imagery, with optional LST information,
      and a trained LCZ_HMSSNet model checkpoint.

      EO raster inputs are provided through a staged STAC catalog directory.
      The trained model checkpoint is provided as a separate file input.
      The processor generates an LCZ classification map GeoTIFF, an optional
      preview PNG, and an output STAC catalog describing the generated product.

    requirements: []

    inputs:

      - id: config
        type: File
        label: prediction configuration
        doc: |
          YAML configuration controlling LCZ inference, including modality,
          model architecture, class ordering, patch size, stride, batch size,
          scaling, nodata handling, and preview generation.

          EO raster paths and model weights are not provided through this
          configuration in the EOAP workflow.

      - id: input_catalog
        type: Directory
        label: input STAC catalog
        doc: |
          Directory containing a STAC catalog named catalog.json referencing
          the staged hyperspectral, Sentinel-2, and optional LST input
          products required for LCZ prediction.

      - id: weights
        type: File
        label: trained model checkpoint
        doc: |
          Trained LCZ_HMSSNet checkpoint used for whole-scene prediction.

      - id: output
        type: string
        label: output LCZ map filename
        default: lcz_map.tif
        doc: |
          Output LCZ classification GeoTIFF path relative to the CWL working
          directory.

    steps:

      processor:
        run: "#lcz_predict_processor"

        in:
          config: config
          input_catalog: input_catalog
          weights: weights
          output: output

        out:
          - output

    outputs:

      output:
        type: Directory
        outputSource: processor/output


  # -------------------------------------------------------------------------
  # LCZ prediction CommandLineTool
  # -------------------------------------------------------------------------
  - class: CommandLineTool
    id: lcz_predict_processor
    label: HEATWISE LCZ Prediction Processor
    doc: |
      HEATWISE LCZ_HMSSNet whole-scene inference processor.

      The processor resolves the EO raster products from the staged STAC
      catalog, loads the supplied trained model checkpoint, performs
      sliding-window LCZ prediction, and writes the resulting classification
      map together with optional preview imagery and an output STAC catalog.

    requirements:

      DockerRequirement:
        dockerPull: ghcr.io/esa-heatwise/heatwise-lcz-classification:eoap-compliance

      InlineJavascriptRequirement: {}

    baseCommand: python

    arguments:
      - /app/processor.py
      - predict

    inputs:

      config:
        type: File
        label: prediction configuration
        doc: |
          YAML configuration controlling the LCZ prediction workflow,
          including modality, model parameters, class order, patch size,
          stride, batch size, scaling, and output options.
        inputBinding:
          prefix: --config

      input_catalog:
        type: Directory
        label: input STAC catalog
        doc: |
          Directory containing catalog.json and the associated STAC Items
          and Assets for the staged hyperspectral, Sentinel-2, and optional
          LST EO products.

          The path to catalog.json inside this directory is passed to the
          processor through the --input-catalog argument.
        inputBinding:
          prefix: --input-catalog
          valueFrom: $(self.path + "/catalog.json")

      weights:
        type: File
        label: trained model checkpoint
        doc: |
          Trained LCZ_HMSSNet checkpoint supplied separately from the EO
          raster input catalog.
        inputBinding:
          prefix: --weights

      output:
        type: string
        label: output LCZ map filename
        default: lcz_map.tif
        doc: |
          Output LCZ classification GeoTIFF path relative to the CWL working
          directory.
        inputBinding:
          prefix: --output

    outputs:

      output:
        type: Directory
        doc: |
          Complete CWL working directory containing all files produced by
          the processor, including the LCZ classification GeoTIFF, optional
          preview PNG, and generated STAC catalog.
        outputBinding:
          glob: "."
