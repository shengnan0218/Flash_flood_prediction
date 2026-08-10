"""Generate deterministic pre-training event and water-level QC artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_quality import build_dataset_quality_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-dir",
        help="默认写入<dataset-root>/qc；只生成审计证据，不重写事件或split",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "qc"
    )
    audit = build_dataset_quality_audit(root)
    written = audit.write(output)
    print(
        json.dumps(
            {"summary": audit.summary, "written": written},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
