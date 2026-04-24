#!/usr/bin/env python3
"""Re-run the v1 LLM judge on a specific list of images to measure test-retest stability.

Usage
-----
    python scripts/llm_judge_rerun.py \
        --files-list path/to/files.txt \
        --species Bombus_ashtoni \
        --image-dir RESULTS_kfold/synthetic_generation \
        --output-path RESULTS_kfold/llm_judge_eval/rerun_results.json

Shares `judge_single_image` and the tier classifier with scripts/llm_judge.py —
this script only exists to run the existing v1 judge on a pre-specified file
list so we can diff verdicts against the original results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from dotenv import load_dotenv

from scripts.llm_judge import judge_single_image  # reuses v1 prompt and schema

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-list", type=Path, required=True)
    ap.add_argument("--species", type=str, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--output-path", type=Path, required=True)
    args = ap.parse_args()

    files = [ln.strip() for ln in args.files_list.read_text().splitlines() if ln.strip()]
    client = OpenAI()

    existing = []
    if args.output_path.exists():
        existing = json.loads(args.output_path.read_text()).get("results", [])
    done = {r["file"] for r in existing}

    results = list(existing)
    for idx, fn in enumerate(files, 1):
        if fn in done:
            continue
        img_path = args.image_dir / args.species / fn
        print(f"[{idx}/{len(files)}] {fn} ...", end=" ", flush=True)
        try:
            verdict = judge_single_image(client, img_path, args.species)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": fn, "species": args.species, "error": str(e), "overall_pass": False})
            continue
        status = "PASS" if verdict["overall_pass"] else "FAIL"
        matches = verdict.get("blind_identification", {}).get("matches_target")
        diag = verdict.get("diagnostic_completeness", {}).get("level")
        print(f"{status}  matches={matches} diag={diag}")
        results.append({"file": fn, "species": args.species, **verdict})
        if idx % 20 == 0:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(json.dumps({"results": results}, indent=2))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps({"results": results}, indent=2))
    passed = sum(1 for r in results if r.get("overall_pass"))
    print(f"\nDone. {passed}/{len(results)} passed. Wrote {args.output_path}")


if __name__ == "__main__":
    main()
