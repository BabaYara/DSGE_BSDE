"""Update data/probab01_table1.json Table 1 values from Tex/Model.tex.

Usage:
  python scripts/update_table1_from_tex.py --tex Tex/Model.tex --json data/probab01_table1.json --dry-run
  python scripts/update_table1_from_tex.py --tex Tex/Model.tex --json data/probab01_table1.json --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from bsde_dsgE.utils.tex_extract import extract_symmetric_states


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tex", type=Path, default=Path("Tex/Model.tex"))
    p.add_argument("--json", type=Path, default=Path("data/probab01_table1.json"))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="Print updated JSON to stdout without writing")
    g.add_argument("--write", action="store_true", help="Write updates back to JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    states = extract_symmetric_states(args.tex)
    data: Dict[str, Any] = json.loads(args.json.read_text())
    data.setdefault("table1_values", {})["symmetric_states"] = states
    payload = json.dumps(data, indent=2)
    if args.write:
        args.json.write_text(payload)
        print({"written": str(args.json), "states": len(states)})
    else:
        print(payload)


if __name__ == "__main__":
    main()

