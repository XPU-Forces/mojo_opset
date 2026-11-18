"""Fetch and verify the task's pinned upstream interpolation reference files."""

import argparse
import hashlib
import json
import urllib.request

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "integrations/vision_pos_embed_interp/reference_sources.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        path = args.output / name
        if path.exists():
            data = path.read_bytes()
        else:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "HiAscend-position-interpolation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != source["sha256"]:
            raise RuntimeError(f"reference checksum mismatch for {name}: {digest}; existing file preserved")
        if not path.exists():
            path.write_bytes(data)
        print(f"{name}: {digest}")
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
