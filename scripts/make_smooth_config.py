#!/usr/bin/env python3
"""
merge_config_fonts.py

Append the "fonts" entries from smooth_config.json onto the
"fonts" array in config_template.json and write the merged JSON to
config.json.

Default filenames:
 - template: config_template.json
 - smooth:   smooth_config.json
 - out:      config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        # Use ensure_ascii=True to preserve \u escapes if present
        json.dump(obj, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


def merge_fonts(template: Dict[str, Any], smooth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new object based on template with smooth['fonts']
    appended to template['fonts'] (creates template['fonts'] if absent).
    """
    merged = dict(template)  # shallow copy of template object
    t_fonts = merged.get("fonts")
    if t_fonts is None:
        t_fonts = []
        merged["fonts"] = t_fonts
    elif not isinstance(t_fonts, list):
        raise TypeError("template 'fonts' must be a JSON array")

    s_fonts = smooth.get("fonts")
    if s_fonts is None:
        # nothing to append
        return merged
    if not isinstance(s_fonts, list):
        raise TypeError("smooth_config 'fonts' must be a JSON array")

    # Append (shallow copy to avoid mutating source)
    t_fonts.extend([dict(item) if isinstance(item, dict) else item
                    for item in s_fonts])
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="merge_config_fonts.py",
        description="Append smooth_config.json fonts into config_template.json"
    )
    p.add_argument("--template", default="config_template.json",
                   help="Template JSON file (default: config_template.json)")
    p.add_argument("--smooth", default="smooth_config.json",
                   help="Smooth JSON file to append (default: smooth_config.json)")
    p.add_argument("--out", default="config.json",
                   help="Output merged JSON (default: config.json)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.template):
        print(f"Template not found: {args.template}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.smooth):
        print(f"Smooth config not found: {args.smooth}", file=sys.stderr)
        return 2

    try:
        template = load_json(args.template)
    except Exception as e:
        print(f"Failed to read template '{args.template}': {e}", file=sys.stderr)
        return 2
    try:
        smooth = load_json(args.smooth)
    except Exception as e:
        print(f"Failed to read smooth config '{args.smooth}': {e}", file=sys.stderr)
        return 2

    if not isinstance(template, dict):
        print("Template JSON must be an object at top level", file=sys.stderr)
        return 2
    if not isinstance(smooth, dict):
        print("Smooth JSON must be an object at top level", file=sys.stderr)
        return 2

    try:
        merged = merge_fonts(template, smooth)
    except Exception as e:
        print(f"Error merging fonts arrays: {e}", file=sys.stderr)
        return 2

    try:
        write_json(args.out, merged)
    except Exception as e:
        print(f"Failed to write output '{args.out}': {e}", file=sys.stderr)
        return 2

    print(f"Wrote merged config to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())