from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import openai
import yaml
from dotenv import load_dotenv

from .assemble import assemble
from .classify import build_reference_payload, classify_all
from .detect import crop_candidates, detect_candidates, load_templates
from .preprocess import preprocess


def run(
    schematic_path: str | Path,
    config_path: str | Path = "config.yaml",
    debug: bool = False,
) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env or environment before running")
    client = openai.OpenAI(api_key=api_key)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["paths"]["output_dir"])
    debug_dir = Path(config["paths"]["debug_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(schematic_path).stem
    print(f"[pipeline] Processing {schematic_path}")

    # Stage 1 — preprocess
    binary = preprocess(schematic_path, config, debug=debug, debug_dir=debug_dir)
    print(f"[pipeline] Stage 1 done — binary shape: {binary.shape}")

    # Stage 2 — detect
    original = cv2.imread(str(schematic_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read schematic: {schematic_path}")

    refs_dir = config["paths"]["refs_dir"]
    templates = load_templates(refs_dir, config["reference_map"])
    boxes = detect_candidates(binary, templates, config, debug=debug, debug_dir=debug_dir, stem=stem)
    print(f"[pipeline] Stage 2 done — {len(boxes)} candidates after NMS")

    crops = crop_candidates(original, boxes)
    print(f"[pipeline] Stage 2b done — {len(crops)} valid crops")

    if not crops:
        print("[pipeline] No candidates found — check detection thresholds")
        return []

    # Stage 3 — classify
    ref_payload = build_reference_payload(refs_dir, config["reference_map"])
    results = classify_all(crops, ref_payload, config, client)
    print(f"[pipeline] Stage 3 done — {len(results)} classifications")

    # Stage 4 — assemble
    annotated, records = assemble(original, crops, results, config)
    print(f"[pipeline] Stage 4 done — {len(records)} valves kept after filtering")

    # Write outputs
    out_img = out_dir / f"{stem}_annotated.png"
    out_json = out_dir / f"{stem}_results.json"

    ok = cv2.imwrite(str(out_img), annotated)
    if not ok:
        print(f"[pipeline] Warning: failed to write annotated image to {out_img}")

    with open(out_json, "w") as f:
        json.dump({"schematic": str(schematic_path), "detections": records}, f, indent=2)

    print(f"[pipeline] Done. {len(records)} valves found.")
    print(f"  Annotated image : {out_img}")
    print(f"  JSON results    : {out_json}")
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P&ID Valve Detection & Classification")
    parser.add_argument("input", nargs="?", help="Path to schematic image")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Save debug images per stage")
    parser.add_argument("--batch", action="store_true", help="Process all images in input/ dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch:
        input_dir = Path(cfg["paths"]["input_dir"])
        schematics = (
            sorted(input_dir.glob("*.png"))
            + sorted(input_dir.glob("*.jpg"))
            + sorted(input_dir.glob("*.tif"))
        )
        if not schematics:
            print(f"[pipeline] No images found in {input_dir}")
        for s in schematics:
            print(f"\n[pipeline] Processing {s.name} ...")
            run(s, config_path=args.config, debug=args.debug)
    elif args.input:
        run(args.input, config_path=args.config, debug=args.debug)
    else:
        parser.print_help()
