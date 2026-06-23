#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sdvg_draft_head_v1 manifests without copying shard files.")
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("manifest_paths", nargs="+")
    args = parser.parse_args()

    output_manifest = Path(args.output_manifest)
    output_root = output_manifest.parent
    merged_shards = []
    total_records = 0
    shard_size = None
    for manifest_arg in args.manifest_paths:
        manifest_path = Path(manifest_arg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "sdvg_draft_head_v1":
            raise ValueError(f"Unsupported manifest format in {manifest_path}: {manifest.get('format')}")
        shard_size = manifest.get("shard_size", shard_size)
        for shard in manifest["shards"]:
            shard_path = manifest_path.parent / shard["path"]
            merged_shards.append(
                {
                    "path": str(shard_path.relative_to(output_root)),
                    "num_records": int(shard["num_records"]),
                }
            )
            total_records += int(shard["num_records"])

    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            {
                "format": "sdvg_draft_head_v1",
                "num_records": total_records,
                "shard_size": shard_size,
                "shards": merged_shards,
                "metadata": {
                    "source_manifests": [str(Path(path).resolve()) for path in args.manifest_paths],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote merged manifest: {output_manifest}")
    print(f"Records: {total_records}")
    print(f"Shards: {len(merged_shards)}")


if __name__ == "__main__":
    main()
