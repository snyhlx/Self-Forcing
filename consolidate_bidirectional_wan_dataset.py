#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate sharded Option-B bidirectional Wan dataset manifests.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--part_dirs", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    total_records = 0
    metadata: dict[str, Any] | None = None
    part_manifests: list[str] = []
    for part_dir_arg in args.part_dirs:
        part_dir = Path(part_dir_arg)
        manifest_path = part_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "bidirectional_wan_full_video_v1":
            raise ValueError(f"Unsupported dataset format in {manifest_path}: {manifest.get('format')}")
        if metadata is None:
            metadata = dict(manifest.get("metadata", {}))
        part_manifests.append(str(manifest_path))
        for shard in manifest.get("shards", []):
            shard_path = (part_dir / shard["path"]).resolve()
            try:
                relative_path = shard_path.relative_to(output_dir.resolve())
            except ValueError:
                relative_path = shard_path
            num_records = int(shard["num_records"])
            shards.append({"path": str(relative_path), "num_records": num_records})
            total_records += num_records

    consolidated = {
        "format": "bidirectional_wan_full_video_v1",
        "num_records": total_records,
        "shard_size": None,
        "shards": shards,
        "metadata": {
            **(metadata or {}),
            "consolidated_from": part_manifests,
            "num_parts": len(args.part_dirs),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    print(f"Wrote consolidated Option-B manifest: {manifest_path}")
    print(f"Records: {total_records} shards: {len(shards)}")


if __name__ == "__main__":
    main()
