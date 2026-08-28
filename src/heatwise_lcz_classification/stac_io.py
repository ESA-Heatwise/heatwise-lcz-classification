from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path


PROCESSOR_NAME = "heatwise-lcz-classification"
PROCESSOR_VERSION = "0.1.1"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_href(item_path: Path, href: str) -> str:
    path = Path(href)

    if path.is_absolute():
        return str(path)

    return str((item_path.parent / path).resolve())


def _catalog_items(catalog_path: str | Path) -> list[tuple[Path, dict]]:
    catalog_path = Path(catalog_path).resolve()

    if not catalog_path.exists():
        raise FileNotFoundError(f"STAC catalog not found: {catalog_path}")

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
            item_path = (catalog_path.parent / item_path).resolve()

        items.append((item_path, _read_json(item_path)))

    if not items:
        raise ValueError(
            f"No STAC Item links found in input catalog: {catalog_path}"
        )

    return items


def patch_h5_from_stac(catalog_path: str | Path) -> str:
    """
    Resolve the patch_h5 asset produced by heatwise-patch-extraction.
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


def prediction_inputs_from_stac(
    catalog_path: str | Path,
    city: str | None = None,
) -> dict:
    """
    Resolve HSI, Sentinel-2 and optional LST inputs for LCZ prediction.
    """
    candidates = []

    for item_path, item in _catalog_items(catalog_path):
        assets = item.get("assets", {})

        if "hsi" not in assets or "sentinel2" not in assets:
            continue

        candidates.append((item_path, item))

    if not candidates:
        raise ValueError(
            "No STAC Item contains both required prediction assets: "
            "`hsi` and `sentinel2`."
        )

    selected_path, selected_item = candidates[0]

    if city:
        city_lower = city.lower()

        for item_path, item in candidates:
            if str(item.get("id", "")).lower() == city_lower:
                selected_path, selected_item = item_path, item
                break

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


def write_training_catalog(
    output_dir: str | Path,
    processor_name: str = PROCESSOR_NAME,
    processor_version: str = PROCESSOR_VERSION,
) -> Path:
    """
    Write a STAC catalog describing the generated training artifacts.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "catalog.json"
    item_filename = "training_artifacts_item.json"
    item_path = output_dir / item_filename

    files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"catalog.json", item_filename}
    ]

    if not files:
        raise ValueError(
            f"No training artifacts found in output directory: {output_dir}"
        )

    assets = {}

    for index, path in enumerate(sorted(files)):
        key = f"artifact_{index + 1}"

        if path.name == "summary.csv":
            key = "summary"

        assets[key] = {
            "href": path.relative_to(output_dir).as_posix(),
            "type": _asset_media_type(path),
            "roles": ["data"],
            "title": path.name,
        }

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
        ],
        "id": "lcz-training-artifacts",
        "geometry": None,
        "properties": {
            "datetime": datetime.now(timezone.utc).isoformat(),
            "processing:software": {
                processor_name: processor_version
            },
        },
        "links": [],
        "assets": assets,
    }

    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
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

    with item_path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2)

    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    return catalog_path


def write_prediction_catalog(
    output_tif: str | Path,
    processor_name: str = PROCESSOR_NAME,
    processor_version: str = PROCESSOR_VERSION,
) -> Path:
    """
    Write a STAC catalog describing the generated LCZ classification map.
    """
    output_tif = Path(output_tif).resolve()
    output_dir = output_tif.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    item_filename = "lcz_prediction_item.json"
    item_path = output_dir / item_filename
    catalog_path = output_dir / "catalog.json"

    assets = {
        "lcz_map": {
            "href": output_tif.name,
            "type": "image/tiff; application=geotiff",
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

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
        ],
        "id": "lcz-prediction",
        "geometry": None,
        "properties": {
            "datetime": datetime.now(timezone.utc).isoformat(),
            "processing:software": {
                processor_name: processor_version
            },
        },
        "links": [],
        "assets": assets,
    }

    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
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

    with item_path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2)

    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    return catalog_path
