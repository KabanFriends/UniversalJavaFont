#!/usr/bin/env python3
"""
make_cell_mapping.py

Free up smooth-font texture cells by sharing identical glyphs, and use the
freed cells to show characters on pages the Bedrock client never loads.

How it works:
 - Bedrock only loads smooth_XX textures for some pages (LOADED_PAGES below,
   kept in sync with BESmoothFontGen's PageUtil). Characters on any other
   page cannot be drawn from their own page.
 - remapping.dat redirects a character to another character's cell; the game
   then uses that cell's texture AND width, and never follows chains.
 - Characters on loaded pages that would render exactly like another loaded
   character (same traced shape, same render parameters, same width) are
   redirected to that character's cell, which frees their own cell.
 - Characters on unloaded pages are then either redirected to an identical
   loaded glyph, or moved into a freed cell (in priority order).

The script mirrors how BESmoothFontGen resolves pixel fonts, traces glyphs and
computes widths. BESmoothFontGen re-checks every shared cell before
generating and refuses to run if any pair would render differently.

Usage:
  python make_cell_mapping.py --config config/ujf.json \
      --priority symbol_chars.json --fill-rest \
      --out config/ujf_cells.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:
    print("This script requires Pillow. Install with: pip install pillow",
          file=sys.stderr)
    raise

from filter_symbols import token_to_chars

# Pages the client loads smooth textures for.
# Keep in sync with BESmoothFontGen's PageUtil.SUPPORTED_PAGES.
LOADED_PAGES = frozenset(
    [0x00, 0x01, 0x02, 0x03, 0x04, 0x20, 0x21, 0x2E, 0x2F]
    + list(range(0x30, 0xD8))
    + list(range(0xF9, 0x100))
)

# ShapeDescGenerator.imageToShapeDesc(): alpha > 25 and not pure black.
SHAPE_ALPHA_THRESHOLD = 25


def f32(x: float) -> float:
    """Round to a 32-bit float, like Java's float arithmetic."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


class PixelFont:
    """Mirror of BESmoothFontGen's PixelWrappedFont for one config entry."""

    def __init__(self, entry: dict, base_dir: str, pixel_dir: str):
        self.name: str = entry["name"]
        self.internal_padding = float(entry.get("padding", 1))
        self.external_padding = float(entry.get("externalPadding", 0))
        self.scale = float(entry.get("scale", 1))
        self.space_width = float(entry.get("spaceWidth", 4))
        self.additional_args: str = entry.get("additionalArgs", "")
        self.overrides: List[Tuple[str, str, float]] = []

        meta = load_json(os.path.join(pixel_dir, self.name + ".json"))
        rows: List[str] = meta["chars"]
        image = Image.open(os.path.join(pixel_dir, self.name + ".png")).convert("RGBA")
        self.grid_w = image.width // len(rows[0])
        self.grid_h = image.height // len(rows)
        self.pixels = image.load()

        # Later occurrences of a character replace earlier ones (HashMap.put).
        self.cells: Dict[str, Tuple[int, int]] = {}
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                self.cells[ch] = (x * self.grid_w, y * self.grid_h)

        override_path = entry.get("widthOverride")
        if override_path:
            full = os.path.join(base_dir, override_path)
            if os.path.isfile(full):
                for ov in load_json(full).get("overrides", []):
                    self.overrides.append((ov["from"][0], ov["to"][0], float(ov["width"])))

    def has_glyph(self, ch: str) -> bool:
        return ch in self.cells

    def _cell_pixels(self, ch: str):
        ox, oy = self.cells[ch]
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                yield x, y, self.pixels[ox + x, oy + y]

    def shape_mask(self, ch: str) -> bytes:
        """Pixels ShapeDescGenerator traces; equal masks give equal shapes."""
        out = bytearray()
        for _x, _y, (r, g, b, a) in self._cell_pixels(ch):
            out.append(1 if a > SHAPE_ALPHA_THRESHOLD and (r + g + b) > 0 else 0)
        return bytes(out)

    def ink_width(self, ch: str) -> int:
        """PixelWrappedFont: last column with any non-zero alpha, max over rows."""
        max_width = 0
        width_by_row: Dict[int, int] = {}
        for x, y, (_r, _g, _b, a) in self._cell_pixels(ch):
            if a != 0:
                width_by_row[y] = x
        for w in width_by_row.values():
            max_width = max(max_width, w)
        return max_width

    def width_inputs(self, ch: str) -> Tuple:
        """Everything PixelWrappedFont.getWidth() + external padding depends on."""
        width = float(self.ink_width(ch))
        if ch == " ":
            width = self.space_width
        for lo, hi, w in self.overrides:
            if lo <= ch <= hi:
                width = w
                break
        return (width, self.internal_padding, self.grid_w, self.scale, self.external_padding)

    def width(self, ch: str) -> float:
        width, pad, grid_w, scale, ext = self.width_inputs(ch)
        w = f32(f32(f32(f32(width + 1) + pad) / grid_w) * scale)
        return f32(w + f32(ext / 64))

    def render_key(self, ch: str) -> Tuple:
        return (
            "pixel", self.grid_w, self.grid_h, self.scale, self.additional_args,
            self.shape_mask(ch), self.width_inputs(ch),
        )


def load_fonts(config: dict, base_dir: str, pixel_dir: str) -> List[PixelFont]:
    fonts: List[PixelFont] = []
    for entry in config.get("fonts", []):
        if entry.get("type") != "pixel":
            raise SystemExit(f"Only pixel fonts are supported, got {entry.get('type')!r} ({entry.get('name') or entry.get('file')})")
        json_path = os.path.join(pixel_dir, entry["name"] + ".json")
        png_path = os.path.join(pixel_dir, entry["name"] + ".png")
        if not (os.path.isfile(json_path) and os.path.isfile(png_path)):
            # BESmoothFontGen skips fonts it cannot load; do the same.
            print(f"Warning: pixel font {entry['name']} not found, skipping", file=sys.stderr)
            continue
        fonts.append(PixelFont(entry, base_dir, pixel_dir))
    if not fonts:
        raise SystemExit("No pixel fonts could be loaded")
    return fonts


def first_font(fonts: Sequence[PixelFont], ch: str) -> Optional[PixelFont]:
    for font in fonts:
        if font.has_glyph(ch):
            return font
    return None  # BESmoothFontGen draws its "missing" glyph here


def read_priority(path: str) -> List[int]:
    """Code points from a symbol_chars.json-style file, in file order."""
    seen = set()
    order: List[int] = []
    for item in load_json(path).get("symbols", []):
        if not isinstance(item, str):
            continue
        for token in item.replace(",", " ").replace(";", " ").replace("|", " ").split():
            for ch in token_to_chars(token):
                cp = ord(ch)
                if cp <= 0xFFFF and cp not in seen:
                    seen.add(cp)
                    order.append(cp)
    return order


def build_mapping(fonts: Sequence[PixelFont], priority: Sequence[int],
                  fill_rest: bool) -> Tuple[List[Tuple[int, int]], dict]:
    # Resolve every character once.
    keys: Dict[int, Tuple] = {}
    for cp in range(1, 0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        font = first_font(fonts, chr(cp))
        if font is not None:
            keys[cp] = font.render_key(chr(cp))

    # Loaded pages: the lowest code point of each identical group keeps its cell.
    canonical: Dict[Tuple, int] = {}
    freeable: List[Tuple[int, int]] = []  # (duplicate cell, canonical cell)
    for cp in sorted(keys):
        if cp >> 8 not in LOADED_PAGES:
            continue
        key = keys[cp]
        if key in canonical:
            freeable.append((cp, canonical[key]))
        else:
            canonical[key] = cp

    # Characters that need a home, in priority order.
    unloaded = [cp for cp in sorted(keys) if cp >> 8 not in LOADED_PAGES]
    unloaded_set = set(unloaded)
    demand: List[int] = [cp for cp in priority if cp in unloaded_set]
    if fill_rest:
        # Skip code points unassigned in this Python's Unicode database: Unifont
        # only has placeholder boxes for them, which are not worth a cell.
        listed = set(demand)
        demand += [cp for cp in unloaded
                   if cp not in listed and unicodedata.category(chr(cp)) != "Cn"]

    mappings: List[Tuple[int, int]] = []
    placed_by_key: Dict[Tuple, int] = {}
    reused = relocated = 0
    unplaced: List[int] = []
    free_iter = iter(freeable)
    for cp in demand:
        key = keys[cp]
        if key in canonical:
            mappings.append((cp, canonical[key]))
            reused += 1
            continue
        if key in placed_by_key:
            mappings.append((cp, placed_by_key[key]))
            reused += 1
            continue
        nxt = next(free_iter, None)
        if nxt is None:
            unplaced.append(cp)
            continue
        cell, cell_canonical = nxt
        mappings.append((cell, cell_canonical))  # free the cell
        mappings.append((cp, cell))             # move the glyph in
        placed_by_key[key] = cell
        relocated += 1

    stats = {
        "loadedGlyphs": sum(1 for cp in keys if cp >> 8 in LOADED_PAGES),
        "freeableCells": len(freeable),
        "usedCells": relocated,
        "unloadedGlyphs": len(unloaded),
        "requested": len(demand),
        "reusedExistingCell": reused,
        "relocated": relocated,
        "unplaced": [f"{cp:04X}" for cp in unplaced],
    }
    return mappings, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True,
                   help="BESmoothFontGen config to read fonts from (updated with cellMapping)")
    p.add_argument("--pixel-dir", default="pixel", help="Directory holding pixel fonts (default: pixel)")
    p.add_argument("--priority", default=None,
                   help="symbol_chars.json-style list of characters to place first")
    p.add_argument("--fill-rest", action="store_true",
                   help="After the priority list, place every other assigned unloaded character "
                        "(in code point order) while cells remain")
    p.add_argument("--out", required=True, help="Cell mapping JSON to write")
    p.add_argument("--report", default=None, help="Optional JSON report path")
    args = p.parse_args(argv)

    base_dir = "."
    config = load_json(args.config)
    if config.get("pageRemapping"):
        print("Error: config still uses pageRemapping; remove it before using cell mapping", file=sys.stderr)
        return 2

    fonts = load_fonts(config, base_dir, args.pixel_dir)
    priority = read_priority(args.priority) if args.priority else []
    mappings, stats = build_mapping(fonts, priority, args.fill_rest)

    write_json(args.out, {"mappings": [{"char": f"{c:04X}", "cell": f"{t:04X}"} for c, t in mappings]})
    config["cellMapping"] = args.out
    write_json(args.config, config)

    print(f"Freeable cells on loaded pages: {stats['freeableCells']}")
    print(f"Requested unloaded characters:  {stats['requested']} "
          f"(reused existing cell: {stats['reusedExistingCell']}, "
          f"moved into freed cell: {stats['relocated']}, "
          f"not placed: {len(stats['unplaced'])})")
    if priority:
        missed = [f"{cp:04X}" for cp in priority if f"{cp:04X}" in set(stats["unplaced"])]
        if missed:
            print(f"Warning: {len(missed)} priority characters could not be placed: {' '.join(missed[:40])}"
                  + (" ..." if len(missed) > 40 else ""), file=sys.stderr)
    print(f"Wrote {len(mappings)} mappings to {args.out}")
    if args.report:
        write_json(args.report, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
