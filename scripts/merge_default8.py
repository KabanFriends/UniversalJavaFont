#!/usr/bin/env python3
"""
merge_default8_consistent.py

Merge pixel-font glyphs into default/default8.png, interpreting translate
values as offsets in the small-glyph (8x8) coordinate space, then scaling
the glyph and translate by:

  s_x = target_tile_w / 8,   s_y = target_tile_h / 8

No fallback is used. Null entries in default8.json clear the tile.

Requires: Pillow (pip install pillow)
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception:
    print("This script requires Pillow. Install with: pip install pillow",
          file=sys.stderr)
    raise

PAGE_GRID = 16
DEFAULT_PNG = "default8.png"
DEFAULT_JSON = "default8.json"
NULL_LITERAL = "\\u0000"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def round_half_up(x: float) -> int:
    """Round half up: 0.5 -> 1, 1.5 -> 2, -0.5 -> 0"""
    return int(math.floor(x + 0.5))


def validate_translate(t: object) -> Tuple[float, float]:
    if not isinstance(t, (list, tuple)) or len(t) != 2:
        raise ValueError("translate must be an array of two numbers")
    return (float(t[0]), float(t[1]))


def is_json_null_entry(s: str) -> bool:
    """True for actual U+0000 or the literal '\\u0000' string."""
    return s == "\x00" or s == NULL_LITERAL


def paste_with_clip(dst: Image.Image, src: Image.Image,
                    dest_left: int, dest_top: int,
                    clip_left: int, clip_top: int,
                    clip_right: int, clip_bottom: int) -> None:
    """
    Paste src into dst at integer position (dest_left, dest_top), but only the
    intersection with [clip_left,clip_top,clip_right,clip_bottom) (dst coords).
    """
    crop_left = max(0, clip_left - dest_left)
    crop_top = max(0, clip_top - dest_top)
    crop_right = crop_left + (clip_right - clip_left)
    crop_bottom = crop_top + (clip_bottom - clip_top)
    src_box = (int(crop_left), int(crop_top), int(crop_right),
               int(crop_bottom))
    src_crop = src.crop(src_box)
    paste_pos = (int(clip_left), int(clip_top))
    dst.paste(src_crop, paste_pos, src_crop)


class PixelFont:
    """
    Pixel font sheet (pixel/<name>.png + pixel/<name>.json).
    Null placeholders ('\\u0000' or U+0000) are ignored.
    """

    def __init__(self, name: str, translate: Tuple[float, float],
                 img: Image.Image, chars_rows: List[str]) -> None:
        self.name = name
        self.translate = translate  # in small-glyph pixels (8x8 space)
        self.image = img.convert("RGBA")
        self.rows = len(chars_rows)
        self.cols = 0 if self.rows == 0 else max(len(r) for r in chars_rows)
        if self.rows == 0 or self.cols == 0:
            raise ValueError(f"Font '{name}' has empty chars array")
        if (self.image.width % self.cols) != 0:
            raise ValueError(
                f"Image width {self.image.width} not divisible by "
                f"{self.cols} cols for font '{name}'"
            )
        if (self.image.height % self.rows) != 0:
            raise ValueError(
                f"Image height {self.image.height} not divisible by "
                f"{self.rows} rows for font '{name}'"
            )
        self.tile_w = self.image.width // self.cols
        self.tile_h = self.image.height // self.rows
        self.char_map: Dict[str, Tuple[int, int]] = {}
        for r, row in enumerate(chars_rows):
            for c, ch in enumerate(row):
                if ch == "\x00" or ch == NULL_LITERAL:
                    continue
                if ch not in self.char_map:
                    self.char_map[ch] = (r, c)

    def crop_char(self, ch: str) -> Optional[Image.Image]:
        pos = self.char_map.get(ch)
        if pos is None:
            return None
        r, c = pos
        left = c * self.tile_w
        top = r * self.tile_h
        return self.image.crop((left, top, left + self.tile_w,
                                top + self.tile_h)).convert("RGBA")


def load_pixel_fonts(config_fonts: List[dict], pixel_dir: str,
                     verbose: bool = True) -> List[PixelFont]:
    fonts: List[PixelFont] = []
    for f in config_fonts:
        name = f.get("name")
        if name is None:
            print("Warning: font entry missing 'name' -> skipping",
                  file=sys.stderr)
            continue
        try:
            tx_ty = validate_translate(f.get("translate"))
        except Exception as e:
            print(f"Warning: font '{name}' invalid translate: {e} -> skip",
                  file=sys.stderr)
            continue
        jpath = os.path.join(pixel_dir, f"{name}.json")
        ppath = os.path.join(pixel_dir, f"{name}.png")
        if not os.path.isfile(jpath):
            print(f"Warning: missing {jpath} -> skipping '{name}'",
                  file=sys.stderr)
            continue
        if not os.path.isfile(ppath):
            print(f"Warning: missing {ppath} -> skipping '{name}'",
                  file=sys.stderr)
            continue
        try:
            font_json = load_json(jpath)
            chars = font_json.get("chars")
            if not isinstance(chars, list):
                raise ValueError("chars must be an array of strings")
            rows = [str(r) for r in chars]
            img = Image.open(ppath)
            pf = PixelFont(name=name, translate=tx_ty, img=img,
                           chars_rows=rows)
            fonts.append(pf)
            if verbose:
                print(f"Loaded pixel font '{name}': {pf.cols}x{pf.rows} "
                      f"tiles, tile={pf.tile_w}x{pf.tile_h}, "
                      f"translate={pf.translate}")
        except Exception as e:
            print(f"Warning: failed loading font '{name}': {e}",
                  file=sys.stderr)
            continue
    return fonts


def merge_default8(default_dir: str, pixel_fonts: List[PixelFont],
                   out_path: str, verbose: bool = False) -> None:
    png_path = os.path.join(default_dir, DEFAULT_PNG)
    json_path = os.path.join(default_dir, DEFAULT_JSON)
    if not os.path.isfile(png_path):
        raise FileNotFoundError(f"Default PNG not found: {png_path}")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Default JSON not found: {json_path}")

    default_im = Image.open(png_path).convert("RGBA")
    default_json = load_json(json_path)
    chars = default_json.get("chars")
    if not isinstance(chars, list):
        raise ValueError(f"{json_path}: missing 'chars' array")
    if len(chars) < PAGE_GRID:
        raise ValueError(f"{json_path}: expected {PAGE_GRID} rows, got "
                         f"{len(chars)}")
    rows = [str(r) for r in chars]
    for i, r in enumerate(rows[:PAGE_GRID]):
        if len(r) < PAGE_GRID:
            raise ValueError(f"{json_path}: row {i} length {len(r)} < "
                             f"{PAGE_GRID} (expected {PAGE_GRID})")

    if (default_im.width % PAGE_GRID) != 0 or \
       (default_im.height % PAGE_GRID) != 0:
        raise ValueError(
            f"{png_path}: image size {default_im.width}x{default_im.height} "
            f"is not divisible by {PAGE_GRID}"
        )

    target_tile_w = default_im.width // PAGE_GRID
    target_tile_h = default_im.height // PAGE_GRID

    sx = float(target_tile_w) / 8.0
    sy = float(target_tile_h) / 8.0

    if verbose:
        print(f"Default sheet: {default_im.width}x{default_im.height}, "
              f"tile={target_tile_w}x{target_tile_h}, scale=({sx:.4f},{sy:.4f})")

    out_im = default_im.copy()
    scaled_cache: Dict[Tuple[str, str], Image.Image] = {}

    for row_idx in range(PAGE_GRID):
        for col_idx in range(PAGE_GRID):
            try:
                ch = rows[row_idx][col_idx]
            except Exception:
                ch = "\x00"

            used_font: Optional[PixelFont] = None
            glyph_scaled: Optional[Image.Image] = None
            tx_pixels = ty_pixels = 0

            for pf in pixel_fonts:
                if ch not in pf.char_map:
                    continue
                used_font = pf
                cache_key = (pf.name, ch)
                if cache_key in scaled_cache:
                    glyph_scaled = scaled_cache[cache_key]
                else:
                    small = pf.crop_char(ch)
                    if small is None:
                        glyph_scaled = None
                    else:
                        # Scale small glyph by sx/sy (small-glyph nominal = 8).
                        sw = max(1, round_half_up(small.width * sx))
                        sh = max(1, round_half_up(small.height * sy))
                        glyph_scaled = small.resize((sw, sh),
                                                   resample=Image.NEAREST)
                        scaled_cache[cache_key] = glyph_scaled
                # Translate is specified in small-glyph pixels; convert using same sx/sy.
                tx_base, ty_base = pf.translate
                tx_pixels = round_half_up(tx_base * sx)
                ty_pixels = round_half_up(ty_base * sy)
                break

            tile_left = col_idx * target_tile_w
            tile_top = row_idx * target_tile_h

            if used_font is not None and glyph_scaled is not None:
                # Erase original tile then paste scaled glyph at computed offset.
                clear = Image.new("RGBA", (target_tile_w, target_tile_h),
                                  (0, 0, 0, 0))
                out_im.paste(clear, (tile_left, tile_top))

                dest_left = tile_left + tx_pixels
                dest_top = tile_top + ty_pixels
                dest_right = dest_left + glyph_scaled.width
                dest_bottom = dest_top + glyph_scaled.height
                clip_left = max(tile_left, dest_left)
                clip_top = max(tile_top, dest_top)
                clip_right = min(tile_left + target_tile_w, dest_right)
                clip_bottom = min(tile_top + target_tile_h, dest_bottom)
                if clip_right <= clip_left or clip_bottom <= clip_top:
                    if verbose:
                        print(f"Skipped (no visible area) '{ord(ch):04X}' "
                              f"from '{used_font.name}' at ({row_idx},{col_idx})")
                    continue
                paste_with_clip(out_im, glyph_scaled, dest_left, dest_top,
                                clip_left, clip_top, clip_right, clip_bottom)
                if verbose:
                    print(f"Placed U+{ord(ch):04X} from '{used_font.name}' "
                          f"at tile ({row_idx},{col_idx}) tx={tx_pixels} ty={ty_pixels} "
                          f"scaled={glyph_scaled.size}")
            else:
                # No pixel glyph found. Clear tile if JSON marks missing, else keep.
                if is_json_null_entry(ch):
                    clear = Image.new("RGBA", (target_tile_w, target_tile_h),
                                      (0, 0, 0, 0))
                    out_im.paste(clear, (tile_left, tile_top))
                    if verbose:
                        print(f"Cleared tile for missing entry at ({row_idx},{col_idx})")
                else:
                    if verbose:
                        try:
                            codepoint = ord(ch)
                            cp_txt = f"U+{codepoint:04X}"
                        except Exception:
                            cp_txt = repr(ch)
                        print(f"Kept default tile for {cp_txt} at ({row_idx},{col_idx})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_im.save(out_path)
    print(f"Wrote merged default sheet: {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="merge_default8_consistent.py",
        description=("Merge pixel fonts into default/default8.png; interpret "
                     "translate in small-glyph (8x8) coordinates and scale "
                     "by target_tile/8."))
    p.add_argument("--config", default="merge_config.json",
                   help="Merger config JSON (default: merge_config.json)")
    p.add_argument("--pixel-dir", default="pixel",
                   help="Directory with pixel fonts (default: pixel)")
    p.add_argument("--default-dir", default="default",
                   help="Directory with default8.png/json (default: default)")
    p.add_argument("--out", default=os.path.join("out", "default8.png"),
                   help="Output path (default: out/default8.png)")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose messages")
    args = p.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_json(args.config)
    cfg_fonts = cfg.get("fonts")
    if not isinstance(cfg_fonts, list):
        print("merge_config.json must contain a 'fonts' array",
              file=sys.stderr)
        return 2

    try:
        pixel_fonts = load_pixel_fonts(cfg_fonts, args.pixel_dir,
                                       verbose=args.verbose)
    except Exception as e:
        print(f"Error loading pixel fonts: {e}", file=sys.stderr)
        return 2

    try:
        merge_default8(args.default_dir, pixel_fonts, args.out,
                       verbose=args.verbose)
    except Exception as e:
        print(f"Error merging default8: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())