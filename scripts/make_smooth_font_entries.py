#!/usr/bin/env python3
"""
Merge config_base.json with per-page unicode files in `pixel/unicode/` directory and
write smooth_config.json with {"fonts": [ ... ]}. Usage:
    python build_smooth_config.py
"""
import argparse
import copy
import json
import os
import re
import sys

PATTERN = re.compile(r'^unicode_([0-9A-Fa-f]{2})\.(?:png|json)$', re.I)

def load_base(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"base file {path} not found")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("config_base.json must contain a JSON object (dictionary)")
    return data

def collect_pages(unicode_dir):
    if not os.path.isdir(unicode_dir):
        raise FileNotFoundError(f"directory {unicode_dir} not found")
    pages = set()
    for fname in os.listdir(unicode_dir):
        m = PATTERN.match(fname)
        if m:
            pages.add(m.group(1).upper())
    return sorted(pages, key=lambda x: int(x, 16))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge unicode files")
    parser.add_argument('--base', '-b', default='config_base.json', help='Path to config_base.json')
    parser.add_argument('--unicode-dir', '-u', default='pixel/unicode', help='Path to unicode directory')
    parser.add_argument('--pixel-prefix', '-p', default='unicode', help='Name prefix of the pixel font')
    parser.add_argument('--out', '-o', default='smooth_config.json', help='Output file path')
    args = parser.parse_args(argv)
    try:
        base = load_base(args.base)
    except Exception as e:
        print(f"Error loading config_base.json: {e}", file=sys.stderr)
        return 2
    try:
        pages = collect_pages(args.unicode_dir)
    except Exception as e:
        print(f"Error collecting unicode files: {e}", file=sys.stderr)
        return 2
    fonts = []
    if not pages:
        print("No unicode_XX files found in the specified directory.", file=sys.stderr)
    for page in pages:
        obj = copy.deepcopy(base)
        obj.update({'name': f'{args.pixel_prefix}/unicode_{page}'})
        fonts.append(obj)
    out = {'fonts': fonts}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(f"Wrote {len(fonts)} entries to {args.out}")

if __name__ == '__main__':
    sys.exit(main())