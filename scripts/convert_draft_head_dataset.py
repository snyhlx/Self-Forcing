#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdvg_draft_head import prepare_target_features_for_storage


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_record(record: dict[str, Any], feature_storage: str) -> dict[str, Any]:
    converted = dict(record)
    converted["target_features"] = prepare_target_features_for_storage(
        record["target_features"],
        storage=feature_storage,
    )
    return converted


def write_manifest(
    *,
    output_dir: Path,
    source_manifest: Path,
    feature_storage: str,
    shard_size: int,
    shards: list[dict[str, Any]],
) -> Path:
    manifest = {
        "format": "sdvg_draft_head_v1",
        "feature_storage": feature_storage,
        "source_manifest": str(source_manifest),
        "num_records": sum(shard["num_records"] for shard in shards),
        "shard_size": shard_size,
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def convert_dataset(
    *,
    source_manifest: Path,
    output_dir: Path,
    feature_storage: str,
    overwrite: bool,
) -> Path:
    source_manifest = source_manifest.resolve()
    source_dir = source_manifest.parent
    output_dir = output_dir.resolve()

    if output_dir == source_dir:
        raise ValueError("output_dir must be different from the source dataset directory")
    if (output_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Output manifest already exists: {output_dir / 'manifest.json'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source_manifest)
    source_shards = manifest["shards"]
    output_shards = []

    for shard_index, shard in enumerate(tqdm(source_shards, desc="convert shards")):
        source_path = source_dir / shard["path"]
        output_name = f"shard_{shard_index:06d}.pt"
        output_path = output_dir / output_name
        if output_path.exists() and overwrite:
            output_path.unlink()
        elif output_path.exists():
            raise FileExistsError(f"Output shard already exists: {output_path}")

        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        records = payload["records"]
        if len(records) != shard["num_records"]:
            raise ValueError(
                f"Shard {shard['path']} expected {shard['num_records']} records, "
                f"found {len(records)}"
            )

        converted_records = [
            convert_record(record, feature_storage=feature_storage)
            for record in tqdm(records, desc=f"records {shard_index:06d}", leave=False)
        ]
        torch.save({"records": converted_records}, output_path)
        output_shards.append({"path": output_name, "num_records": len(converted_records)})

        del payload, records, converted_records
        gc.collect()

    return write_manifest(
        output_dir=output_dir,
        source_manifest=source_manifest,
        feature_storage=feature_storage,
        shard_size=manifest.get("shard_size", 0),
        shards=output_shards,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert draft-head dataset feature storage.")
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_storage", choices=["pooled", "full"], default="pooled")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = convert_dataset(
        source_manifest=Path(args.source_manifest),
        output_dir=Path(args.output_dir),
        feature_storage=args.feature_storage,
        overwrite=args.overwrite,
    )
    print(f"Wrote converted manifest: {manifest_path}")
    print("Delete original after verifying the converted dataset with:")
    print(
        "rm -rf "
        "/root/Self-Forcing/outputs/sdvg/draft_head_dataset/moviegen_target_context"
    )


if __name__ == "__main__":
    main()
