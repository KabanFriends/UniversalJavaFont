#!/usr/bin/env python3
"""
Build unicode_XX.png and unicode_XX.json pages from Unifont .hex files.

Directory layout expected (relative to running directory):
 - unifont_list.json
 - unifont_hex/      (contains .hex files named like unifont.hex, etc.)
 - pixel/unicode/         (will be created if missing)

See --help for options.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - dependency check
    print("This script requires Pillow. Install with: pip install pillow")
    raise


HEX_LINE_RE = re.compile(r"^\s*([0-9A-Fa-f]+)\s*:\s*([0-9A-Fa-f]+)\s*$")


def parse_unifont_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    fonts = obj.get("fonts")
    if not isinstance(fonts, list):
        raise ValueError("unifont_list.json must contain a 'fonts' array")
    return [str(x) for x in fonts]


def find_hex_path(hex_dir: str, name: str) -> Optional[str]:
    # Accept names with or without ".hex"
    candidates = []
    if name.lower().endswith(".hex"):
        candidates.append(os.path.join(hex_dir, name))
    else:
        candidates.append(os.path.join(hex_dir, f"{name}.hex"))
        candidates.append(os.path.join(hex_dir, name))  # just in case
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_hex_fonts(font_names: Iterable[str], hex_dir: str) -> Dict[int, str]:
    """
    Load .hex files in the order of font_names. Earlier names take
    precedence; we only add a mapping for a codepoint the first time
    we encounter it (so earlier fonts win).
    Returns mapping: codepoint (int) -> bitmap hex string.
    """
    mapping: Dict[int, str] = {}
    for fname in font_names:
        path = find_hex_path(hex_dir, fname)
        if not path:
            print(f"Warning: .hex file for '{fname}' not found in {hex_dir}", 
                  file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = HEX_LINE_RE.match(line)
                if not m:
                    # skip lines that don't match; warn for debugging
                    continue
                cp_hex, bmp_hex = m.group(1), m.group(2)
                try:
                    cp = int(cp_hex, 16)
                except ValueError:
                    continue
                if cp < 0x0000 or cp > 0xFFFF:
                    # out of BMP range for this script
                    continue
                bmp_len = len(bmp_hex)
                if bmp_len not in (32, 64):
                    # ignore non-32/64 entries
                    continue
                # precedence: only add when not already present
                if cp not in mapping:
                    mapping[cp] = bmp_hex.lower()
    return mapping


def decode_hex_to_grid(bmp_hex: str) -> List[List[int]]:
    """
    Convert a 32- or 64-hex-digit glyph into a 16x16 grid of 0/1 ints.
    For 32 hex digits -> width 8, left-aligned into 16 columns.
    For 64 hex digits -> width 16.
    """
    bmp_len = len(bmp_hex)
    rows: List[List[int]] = [[0] * 16 for _ in range(16)]
    if bmp_len == 32:
        # 2 hex digits per row -> 16 rows
        for r in range(16):
            row_hex = bmp_hex[r * 2 : r * 2 + 2]
            byte = int(row_hex, 16)
            for c in range(8):
                bit = (byte >> (7 - c)) & 1
                rows[r][c] = 1 if bit else 0
    elif bmp_len == 64:
        # 4 hex digits per row (2 bytes) -> 16 rows * 16 columns
        for r in range(16):
            row_hex = bmp_hex[r * 4 : r * 4 + 4]
            hi = int(row_hex[0:2], 16)
            lo = int(row_hex[2:4], 16)
            for c in range(8):
                if ((hi >> (7 - c)) & 1):
                    rows[r][c] = 1
            for c in range(8):
                if ((lo >> (7 - c)) & 1):
                    rows[r][8 + c] = 1
    else:
        raise ValueError("bmp_hex must be 32 or 64 hex digits long")
    return rows


def parse_color(spec: str) -> Tuple[int, int, int, int]:
    """
    Accepts:
     - named 'white'/'black'
     - hex '#RRGGBB' or 'RRGGBB'
     - 'R,G,B' or 'R,G,B,A' decimal
    Returns (R,G,B,A)
    """
    s = spec.strip().lower()
    if s in ("white", "#fff", "fff"):
        return (255, 255, 255, 255)
    if s in ("black", "#000", "000"):
        return (0, 0, 0, 255)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(ch in "0123456789abcdef" for ch in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b, 255)
    parts = [p.strip() for p in spec.split(",")]
    if 3 <= len(parts) <= 4:
        vals = [int(p) for p in parts]
        if len(vals) == 3:
            vals.append(255)
        return (vals[0], vals[1], vals[2], vals[3])
    raise ValueError(f"Cannot parse color spec: {spec}")


def build_page_image_and_json(page: int, mapping: Dict[int, str],
                              out_dir: str, fg: Tuple[int, int, int, int],
                              scale: int = 1) -> None:
    """
    Build unicode_XX.png and unicode_XX.json for given page 0..255.
    """
    page_hex = f"{page:02X}"
    img_size = 16 * 16 * scale  # 256 * scale
    img = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
    pixels = img.load()

    chars_rows: List[str] = []
    for row in range(16):
        row_chars = []
        for col in range(16):
            low = row * 16 + col
            cp = (page << 8) | low
            if cp in mapping:
                row_chars.append(chr(cp))
                grid = decode_hex_to_grid(mapping[cp])
                # draw grid into the big image, with optional scale
                tx = col * 16 * scale
                ty = row * 16 * scale
                for r in range(16):
                    for c in range(16):
                        if grid[r][c]:
                            # draw scaled pixel
                            for sy in range(scale):
                                for sx in range(scale):
                                    x = tx + c * scale + sx
                                    y = ty + r * scale + sy
                                    pixels[x, y] = fg
            else:
                # missing glyph -> write U+0000 char as placeholder
                row_chars.append(chr(0))
                # leave pixels transparent
        chars_rows.append("".join(row_chars))

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"unicode_{page_hex}.png")
    json_path = os.path.join(out_dir, f"unicode_{page_hex}.json")
    img.save(png_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        # ensure_ascii=True to get \uXXXX escapes (including \u0000)
        json.dump({"chars": chars_rows}, fh, ensure_ascii=True, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="build_unifont_pages.py",
        description="Create unicode_XX.png/json pages from Unifont .hex files",
    )
    p.add_argument("--list-file", default="unifont_list.json",
                   help="JSON file listing fonts (default: unifont_list.json)")
    p.add_argument("--hex-dir", default="unifont_hex",
                   help="Directory containing .hex files (default: unifont_hex)")
    p.add_argument("--out-dir", default="pixel/unicode",
                   help="Output directory for unicode_XX.png/json (default: pixel/unicode)")
    p.add_argument("--fg", default="white",
                   help="Foreground color (name, #RRGGBB, or R,G,B[,A])")
    p.add_argument("--scale", type=int, default=1,
                   help="Integer scale factor for output PNGs (default 1)")
    p.add_argument("--pages", default="all",
                   help="Comma-separated page numbers (hex) or 'all' (default). "
                        "Examples: '00,01,02' or '00-0F' or 'all'")
    args = p.parse_args(argv)

    try:
        fg_color = parse_color(args.fg)
    except Exception as e:
        print(f"Error parsing --fg: {e}", file=sys.stderr)
        return 2
    if args.scale < 1:
        print("--scale must be >= 1", file=sys.stderr)
        return 2

    try:
        font_names = parse_unifont_list(args.list_file)
    except Exception as e:
        print(f"Error reading {args.list_file}: {e}", file=sys.stderr)
        return 2

    mapping = load_hex_fonts(font_names, args.hex_dir)
    print(f"Loaded {len(mapping)} glyphs from {args.hex_dir} (precedence order).")

    # determine pages
    pages_to_build: List[int] = []
    if args.pages.strip().lower() == "all":
        pages_to_build = list(range(256))
    else:
        s = args.pages.strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                start = int(a, 16)
                end = int(b, 16)
                pages_to_build.extend(range(start, end + 1))
            else:
                pages_to_build.append(int(part, 16))

    # unique & clamp to 0..255
    pages = sorted({p & 0xFF for p in pages_to_build})
    for page in pages:
        print(f"Building page {page:02X} ...")
        build_page_image_and_json(page, mapping, args.out_dir, fg_color,
                                  scale=args.scale)

    print(f"All done. Output in: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())