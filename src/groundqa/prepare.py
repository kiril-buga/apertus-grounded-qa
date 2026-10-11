"""Build train/val/test JSONL files once, so every later step sees the same data.

    python -m groundqa.prepare --config configs/apertus8b.yaml
"""

import json
from pathlib import Path

from .config import base_parser, load_config, write_jsonl
from .data import prepare_splits, summarize


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    out_dir = Path(cfg["data"]["dir"])

    splits = prepare_splits(cfg)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    (out_dir / "data_config.json").write_text(json.dumps(cfg["data"], indent=2))

    print(summarize(splits))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
