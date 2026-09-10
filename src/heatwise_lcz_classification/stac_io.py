from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


PROCESSOR_NAME = "heatwise-lcz-classification"
PROCESSOR_VERSION = "0.1.1"
STAC_VERSION = "1.1.0"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_href(item_path: Path, href: str) -> str:
    path = Path(href)

    if path.is_absolute():
        return str(path)

    return str((item_path.parent / path).resolve())


def _catalog_items(
    catalog_path: str | Path,
) -> list[tuple[Path, dict]]:
    catalog_path = Path(catalog_path).resolve()

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"STAC catalog not found: {catalog_path}"
        )

    catalog = _read_json(catalog_path)
    items = []

    for link in catalog.get("links", []):
        if link.get("rel") != "item":
            continue

        href = link.get("href")

        if not href:
            continue

        item_path = Path(href)

        if not item_path.is_absolute():
            item_path = (
                catalog_path.parent / item_path
            ).resolve()

        items.append(
            (
                item_path,
                _read_json(item_path),
            )
        )

    if not items:
        raise ValueError(
            "No STAC Item links found in input catalog: "
            f"{catalog_path}"
        )

    return items


def _item_metadata(item: dict) -> dict:
    """
    Extract spatial and temporal metadata from a STAC Item.
    """
    properties = item.get("properties", {})

    temporal = {}

    if "datetime" in properties:
        temporal["datetime"] = properties.get("datetime")

    if "start_datetime" in properties:
        temporal["start_datetime"] = properties.get(
            "start_datetime"
        )

    if "end_datetime" in properties:
        temporal["end_datetime"] = properties.get(
            "end_datetime"
        )

    return {
        "geometry": item.get("geometry"),
        "bbox": item.get("bbox"),
        "properties": temporal,
    }


def _temporal_properties(
    metadata: dict | None,
) -> dict:
    """
    Preserve temporal metadata from the input STAC Item.

    Prefer a non-null datetime. If the source represents an
    interval, preserve datetime=None together with
    start_datetime/end_datetime.

    Fall back to processing time only when no source temporal
    metadata are available.
    """
    metadata = metadata or {}
    properties = metadata.get("properties", {}) or {}

    source_datetime = properties.get("datetime")

    if source_datetime:
        return {
            "datetime": source_datetime,
        }

    start_datetime = properties.get("start_datetime")
    end_datetime = properties.get("end_datetime")

    if start_datetime or end_datetime:
        temporal = {
            "datetime": None,
        }

        if start_datetime:
            temporal["start_datetime"] = start_datetime

        if end_datetime:
            temporal["end_datetime"] = end_datetime

        return temporal

    return {
        "datetime": datetime.now(timezone.utc).isoformat(),
    }


def patch_h5_from_stac(
    catalog_path: str | Path,
) -> str:
    """
    Resolve the patch_h5 asset produced by
    heatwise-patch-extraction.
    """
    for item_path, item in _catalog_items(catalog_path):
        assets = item.get("assets", {})

        if "patch_h5" in assets:
            return _resolve_href(
                item_path,
                assets["patch_h5"]["href"],
            )

    raise ValueError(
        "No STAC Item contains the required `patch_h5` asset."
    )


def patch_h5_metadata_from_stac(
    catalog_path: str | Path,
) -> dict:
    """
    Return spatial and temporal metadata from the STAC Item
    containing the patch_h5 training asset.
    """
    for _, item in _catalog_items(catalog_path):
        assets = item.get("assets", {})

        if "patch_h5" in assets:
            return _item_metadata(item)

    raise ValueError(
        "No STAC Item contains the required `patch_h5` asset."
    )


def _prediction_candidates(
    catalog_path: str | Path,
) -> list[tuple[Path, dict]]:
    """
    Return STAC Items containing the required prediction assets.
    """
    candidates = []

    for item_path, item in _catalog_items(catalog_path):
        assets = item.get("assets", {})

        if (
            "hsi" not in assets
            or "sentinel2" not in assets
        ):
            continue

        candidates.append(
            (
                item_path,
                item,
            )
        )

    if not candidates:
        raise ValueError(
            "No STAC Item contains both required prediction "
            "assets: `hsi` and `sentinel2`."
        )

    return candidates


def _select_prediction_item(
    catalog_path: str | Path,
    city: str | None = None,
) -> tuple[Path, dict]:
    candidates = _prediction_candidates(catalog_path)

    selected_path, selected_item = candidates[0]

    if city:
        city_lower = city.lower()

        for item_path, item in candidates:
            if (
                str(item.get("id", "")).lower()
                == city_lower
            ):
                selected_path = item_path
                selected_item = item
                break

    return selected_path, selected_item


def prediction_inputs_from_stac(
    catalog_path: str | Path,
    city: str | None = None,
) -> dict:
    """
    Resolve HSI, Sentinel-2 and optional LST inputs for
    LCZ prediction.
    """
    selected_path, selected_item = (
        _select_prediction_item(
            catalog_path,
            city=city,
        )
    )

    assets = selected_item["assets"]

    inputs = {
        "hsi": _resolve_href(
            selected_path,
            assets["hsi"]["href"],
        ),
        "sen2": _resolve_href(
            selected_path,
            assets["sentinel2"]["href"],
        ),
    }

    if "lst" in assets:
        inputs["lst"] = _resolve_href(
            selected_path,
            assets["lst"]["href"],
        )

    return inputs


def prediction_metadata_from_stac(
    catalog_path: str | Path,
    city: str | None = None,
) -> dict:
    """
    Return spatial and temporal metadata from the STAC Item
    selected for LCZ prediction.
    """
    _, selected_item = _select_prediction_item(
        catalog_path,
        city=city,
    )

    return _item_metadata(selected_item)


def _asset_media_type(path: Path) -> str:
    suffix = path.suffix.lower()

    explicit_types = {
        ".pth": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".csv": "text/csv",
        ".tif": "image/tiff; application=geotiff",
        ".tiff": "image/tiff; application=geotiff",
        ".png": "image/png",
        ".json": "application/json",
    }

    if suffix in explicit_types:
        return explicit_types[suffix]

    guessed, _ = mimetypes.guess_type(path.name)

    return guessed or "application/octet-stream"


def _raster_spatial_metadata(
    raster_path: str | Path,
) -> tuple[dict, list[float]]:
    """
    Derive GeoJSON geometry and bbox in EPSG:4326 from a
    georeferenced output raster.
    """
    raster_path = Path(raster_path).resolve()

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Output raster not found: {raster_path}"
        )

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(
                "Cannot create meaningful STAC geometry because "
                f"the raster has no CRS: {raster_path}"
            )

        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21,
        )

    bbox = [
        west,
        south,
        east,
        north,
    ]

    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }

    return geometry, bbox


def write_training_catalog(
    output_dir: str | Path,
    input_metadata: dict | None = None,
    processor_name: str = PROCESSOR_NAME,
    processor_version: str = PROCESSOR_VERSION,
) -> Path:
    """
    Write a STAC catalog describing the generated training
    artifacts.

    Training artifacts themselves are not geospatial rasters.
    Their spatial and temporal extent therefore comes from the
    patch dataset used for training.
    """
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog_path = output_dir / "catalog.json"
    item_filename = "training_artifacts_item.json"
    item_path = output_dir / item_filename

    files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "catalog.json",
            item_filename,
        }
    ]

    if not files:
        raise ValueError(
            "No training artifacts found in output "
            f"directory: {output_dir}"
        )

    assets = {}

    for index, path in enumerate(sorted(files)):
        key = f"artifact_{index + 1}"

        if path.name == "summary.csv":
            key = "summary"

        assets[key] = {
            "href": (
                path.relative_to(output_dir).as_posix()
            ),
            "type": _asset_media_type(path),
            "roles": ["data"],
            "title": path.name,
        }

    input_metadata = input_metadata or {}

    geometry = input_metadata.get("geometry")
    bbox = input_metadata.get("bbox")

    properties = _temporal_properties(
        input_metadata
    )

    properties["processing:software"] = {
        processor_name: processor_version
    }

    item = {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            (
                "https://stac-extensions.github.io/"
                "processing/v1.2.0/schema.json"
            )
        ],
        "id": "lcz-training-artifacts",
        "geometry": geometry,
        "properties": properties,
        "links": [],
        "assets": assets,
    }

    if bbox is not None:
        item["bbox"] = bbox

    catalog = {
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "id": f"{processor_name}-training-output",
        "description": (
            "HEATWISE LCZ classification training artifacts."
        ),
        "links": [
            {
                "rel": "item",
                "href": item_filename,
                "type": "application/geo+json",
            }
        ],
    }

    with item_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            item,
            f,
            indent=2,
        )

    with catalog_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            catalog,
            f,
            indent=2,
        )

    return catalog_path


def write_prediction_catalog(
    output_tif: str | Path,
    input_metadata: dict | None = None,
    processor_name: str = PROCESSOR_NAME,
    processor_version: str = PROCESSOR_VERSION,
) -> Path:
    """
    Write a STAC catalog describing the generated LCZ
    classification map.

    Geometry and bbox are derived from the actual output GeoTIFF.
    Temporal metadata are propagated from the input STAC Item.
    """
    output_tif = Path(output_tif).resolve()
    output_dir = output_tif.parent

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output_tif.exists():
        raise FileNotFoundError(
            f"LCZ output map not found: {output_tif}"
        )

    item_filename = "lcz_prediction_item.json"
    item_path = output_dir / item_filename
    catalog_path = output_dir / "catalog.json"

    assets = {
        "lcz_map": {
            "href": output_tif.name,
            "type": (
                "image/tiff; application=geotiff"
            ),
            "roles": ["data"],
            "title": "LCZ classification map",
        }
    }

    preview = output_tif.with_name(
        f"{output_tif.stem}_preview.png"
    )

    if preview.exists():
        assets["preview"] = {
            "href": preview.name,
            "type": "image/png",
            "roles": ["overview"],
            "title": "LCZ classification preview",
        }

    geometry, bbox = _raster_spatial_metadata(
        output_tif
    )

    properties = _temporal_properties(
        input_metadata
    )

    properties["processing:software"] = {
        processor_name: processor_version
    }

    item = {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            (
                "https://stac-extensions.github.io/"
                "processing/v1.2.0/schema.json"
            )
        ],
        "id": "lcz-prediction",
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "links": [],
        "assets": assets,
    }

    catalog = {
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "id": f"{processor_name}-prediction-output",
        "description": (
            "HEATWISE LCZ classification prediction output."
        ),
        "links": [
            {
                "rel": "item",
                "href": item_filename,
                "type": "application/geo+json",
            }
        ],
    }

    with item_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            item,
            f,
            indent=2,
        )

    with catalog_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            catalog,
            f,
            indent=2,
        )

    return catalog_path
