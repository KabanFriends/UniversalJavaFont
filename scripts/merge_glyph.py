#!/usr/bin/env python3
"""
merge_fonts_with_width_overrides.py

Merge pixel fonts into Unifont pages and optionally mark widths for ranges
defined in unifont_width_overrides.json.

Behavior additions:
 - Loads unifont_width_overrides.json (optional). Format:
   {
     "overrides": [
       { "from": "\u3001", "to": "\u30FF", "width": 16 },
       ...
     ]
   }
 - For codepoints inside an override range, if the glyph remains the
   Unifont glyph (i.e. not replaced by a pixel glyph or fallback), place
   a single barely-visible pixel at coordinate (width,16) in the 1-indexed
   16×16 tile coordinate system (converted to final pixel coordinates).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:
    print("This script requires Pillow. Install with: pip install pillow",
          file=sys.stderr)
    raise

PAGE_PNG_RE = re.compile(r"^unicode_([0-9A-Fa-f]{2})\.png$")


def round_half_up(x: float) -> int:
    """Round-half-up to avoid banker's rounding issues."""
    return int(math.floor(x + 0.5))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_cp_token(token: object) -> Optional[int]:
    """
    Parse a token that may represent a codepoint or a single character.
    Accept several forms:
      - single-character string -> ord(...)
      - escape forms like "\\u3001" (will be decoded)
      - "U+3001", "u3001", "0x3001", "3001" (hex)
      - strings like "/u3001" (we extract hex digits)
    Returns codepoint int or None on failure.
    """
    if token is None:
        return None
    if isinstance(token, int):
        return token if 0 <= token <= 0x10FFFF else None
    s = str(token)
    if not s:
        return None
    # Single char (already decoded): use its codepoint
    if len(s) == 1:
        return ord(s)
    # Try to decode unicode-escape sequences first
    try:
        dec = s.encode("utf-8").decode("unicode_escape")
        if dec:
            # take the first character after unescaping
            return ord(dec[0])
    except Exception:
        pass
    # Try common hex forms
    m = re.search(r"[0-9A-Fa-f]{1,6}", s)
    if m:
        try:
            v = int(m.group(0), 16)
            if 0 <= v <= 0x10FFFF:
                return v
        except Exception:
            pass
    return None


def load_width_overrides(path: str) -> List[Tuple[int, int, int]]:
    """
    Load overrides file and return list of (from_cp, to_cp, width).
    Ranges inclusive. Widths are clamped to 1..16.
    If the file does not exist, returns empty list.
    """
    if not os.path.isfile(path):
        return []
    try:
        obj = load_json(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load overrides '{path}': {e}")
    arr = obj.get("overrides")
    if not isinstance(arr, list):
        raise RuntimeError(f"{path}: 'overrides' must be an array")
    out: List[Tuple[int, int, int]] = []
    for i, ent in enumerate(arr):
        if not isinstance(ent, dict):
            print(f"Warning: skip invalid override entry {i}", file=sys.stderr)
            continue
        from_tok = ent.get("from")
        to_tok = ent.get("to", from_tok)
        width_val = ent.get("width")
        if width_val is None:
            print(f"Warning: override entry {i} missing 'width' -> skip",
                  file=sys.stderr)
            continue
        try:
            w = int(width_val)
        except Exception:
            print(f"Warning: override entry {i} invalid width -> skip",
                  file=sys.stderr)
            continue
        cp_from = parse_cp_token(from_tok)
        cp_to = parse_cp_token(to_tok)
        if cp_from is None or cp_to is None:
            print(f"Warning: override entry {i} invalid from/to -> skip",
                  file=sys.stderr)
            continue
        if cp_from > cp_to:
            cp_from, cp_to = cp_to, cp_from
        # clamp width to 1..16 (tile coordinate space)
        if w < 1:
            w = 1
        if w > 16:
            w = 16
        out.append((cp_from, cp_to, w))
    return out


def find_override_for_cp(ranges: Sequence[Tuple[int, int, int]],
                         cp: int) -> Optional[int]:
    """Return width if cp falls within any override range, else None."""
    for a, b, w in ranges:
        if a <= cp <= b:
            return w
    return None


def write_png(im: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)


def validate_translate(t: object) -> Tuple[float, float]:
    if (not isinstance(t, (list, tuple))) or len(t) != 2:
        raise ValueError("translate must be an array of two numbers")
    return (float(t[0]), float(t[1]))


class PixelFont:
    def __init__(self, name: str, translate: Tuple[float, float],
                 img: Image.Image, chars_rows: List[str]) -> None:
        self.name = name
        self.translate = translate
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
                if ch == "\x00" or ch == "\\u0000":
                    # skip null placeholders (so fallback can be used)
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
        translate = f.get("translate")
        try:
            tx_ty = validate_translate(translate)
        except Exception as e:
            print(f"Warning: font '{name}' invalid translate: {e} -> skip",
                  file=sys.stderr)
            continue
        json_path = os.path.join(pixel_dir, f"{name}.json")
        png_path = os.path.join(pixel_dir, f"{name}.png")
        if not os.path.isfile(json_path):
            print(f"Warning: missing {json_path} -> skipping font '{name}'",
                  file=sys.stderr)
            continue
        if not os.path.isfile(png_path):
            print(f"Warning: missing {png_path} -> skipping font '{name}'",
                  file=sys.stderr)
            continue
        try:
            font_json = load_json(json_path)
            chars = font_json.get("chars")
            if not isinstance(chars, list):
                raise ValueError("chars must be an array of strings")
            rows = [str(r) for r in chars]
            img = Image.open(png_path)
            pf = PixelFont(name=name, translate=tx_ty, img=img,
                           chars_rows=rows)
            fonts.append(pf)
            if verbose:
                print(f"Loaded pixel font '{name}': {pf.cols}x{pf.rows} tiles, "
                      f"tile={pf.tile_w}x{pf.tile_h}, translate={pf.translate}")
        except Exception as e:
            print(f"Warning: failed loading font '{name}': {e}",
                  file=sys.stderr)
            continue
    return fonts


def load_fallback(png_path: str) -> Optional[Image.Image]:
    if not os.path.isfile(png_path):
        return None
    return Image.open(png_path).convert("RGBA")


def list_unicode_pages(unicode_dir: str) -> List[Tuple[int, str, str]]:
    entries: List[Tuple[int, str, str]] = []
    if not os.path.isdir(unicode_dir):
        raise FileNotFoundError(f"Unicode dir not found: {unicode_dir}")
    for name in os.listdir(unicode_dir):
        m = PAGE_PNG_RE.match(name)
        if not m:
            continue
        page_hex = m.group(1).upper()
        png_path = os.path.join(unicode_dir, name)
        json_name = f"unicode_{page_hex}.json"
        json_path = os.path.join(unicode_dir, json_name)
        if not os.path.isfile(json_path):
            print(f"Warning: missing JSON for page {page_hex}: {json_path}",
                  file=sys.stderr)
            continue
        try:
            page_int = int(page_hex, 16)
        except Exception:
            continue
        entries.append((page_int, png_path, json_path))
    entries.sort(key=lambda x: x[0])
    return entries


def paste_with_clip(dst: Image.Image, src: Image.Image,
                    dest_left: int, dest_top: int,
                    clip_left: int, clip_top: int,
                    clip_right: int, clip_bottom: int) -> None:
    crop_left = max(0, clip_left - dest_left)
    crop_top = max(0, clip_top - dest_top)
    crop_right = crop_left + (clip_right - clip_left)
    crop_bottom = crop_top + (clip_bottom - clip_top)
    src_box = (int(crop_left), int(crop_top), int(crop_right),
               int(crop_bottom))
    src_crop = src.crop(src_box)
    paste_pos = (int(clip_left), int(clip_top))
    dst.paste(src_crop, paste_pos, src_crop)


def is_json_null_entry(s: str) -> bool:
    return s == "\x00" or s == "\\u0000"


def merge_page(page: int, png_path: str, json_path: str,
               pixel_fonts: List[PixelFont],
               fallback_img: Optional[Image.Image],
               fallback_translate: Tuple[float, float],
               out_dir: str,
               width_overrides: Sequence[Tuple[int, int, int]],
               verbose: bool = False) -> None:
    page_hex = f"{page:02X}"
    if verbose:
        print(f"Processing page {page_hex} ...")
    uni_im = Image.open(png_path).convert("RGBA")
    uni_json = load_json(json_path)
    chars = uni_json.get("chars")
    if not isinstance(chars, list):
        raise ValueError(f"{json_path}: missing 'chars' array")
    if len(chars) < 16:
        raise ValueError(f"{json_path}: expected 16 rows, got {len(chars)}")
    rows = [str(r) for r in chars]
    for i, r in enumerate(rows[:16]):
        if len(r) < 16:
            raise ValueError(
                f"{json_path}: row {i} length {len(r)} < 16 (expected 16)"
            )

    if (uni_im.width % 16) != 0 or (uni_im.height % 16) != 0:
        raise ValueError(
            f"{png_path}: image size {uni_im.width}x{uni_im.height} "
            "is not divisible by 16"
        )
    target_tile_w = uni_im.width // 16
    target_tile_h = uni_im.height // 16
    page_scale_x = target_tile_w / 16.0
    page_scale_y = target_tile_h / 16.0

    # constants for marker placement: bottom row index (1-indexed 16 ->
    # 0-based index 15)
    bottom_index_0 = 16 - 1

    out_im = uni_im.copy()
    scaled_cache: Dict[Tuple[str, str], Image.Image] = {}
    fallback_scaled: Optional[Image.Image] = None

    for row in range(16):
        for col in range(16):
            cp = (page << 8) | (row * 16 + col)
            ch = chr(cp)
            used_font: Optional[PixelFont] = None
            glyph_scaled: Optional[Image.Image] = None
            tx_pixels = ty_pixels = 0

            # try to find a pixel font glyph
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
                        sx = 2.0 * page_scale_x
                        sy = 2.0 * page_scale_y
                        sw = max(1, round_half_up(small.width * sx))
                        sh = max(1, round_half_up(small.height * sy))
                        glyph_scaled = small.resize((sw, sh),
                                                   resample=Image.NEAREST)
                        scaled_cache[cache_key] = glyph_scaled
                tx_base, ty_base = pf.translate
                tx_pixels = round_half_up(tx_base * (2.0 * page_scale_x))
                ty_pixels = round_half_up(ty_base * (2.0 * page_scale_y))
                break

            tile_left = col * target_tile_w
            tile_top = row * target_tile_h

            if used_font is not None:
                # Erase tile and paste pixel glyph
                clear = Image.new("RGBA", (target_tile_w, target_tile_h),
                                  (0, 0, 0, 0))
                out_im.paste(clear, (tile_left, tile_top))

                dest_left = tile_left + tx_pixels
                dest_top = tile_top + ty_pixels
                dest_right = dest_left + glyph_scaled.width \
                    if glyph_scaled is not None else dest_left
                dest_bottom = dest_top + glyph_scaled.height \
                    if glyph_scaled is not None else dest_top
                clip_left = max(tile_left, dest_left)
                clip_top = max(tile_top, dest_top)
                clip_right = min(tile_left + target_tile_w, dest_right)
                clip_bottom = min(tile_top + target_tile_h, dest_bottom)
                if clip_right <= clip_left or clip_bottom <= clip_top:
                    # nothing visible
                    continue
                paste_with_clip(out_im, glyph_scaled, dest_left, dest_top,
                                clip_left, clip_top, clip_right, clip_bottom)
                continue

            # No pixel font glyph found. Decide if fallback used or Unifont remains.
            try:
                uchar = rows[row][col]
            except Exception:
                uchar = "\x00"

            # fallback case
            if is_json_null_entry(uchar) and fallback_img is not None:
                if fallback_scaled is None:
                    sx = 2.0 * page_scale_x
                    sy = 2.0 * page_scale_y
                    fw = max(1, round_half_up(fallback_img.width * sx))
                    fh = max(1, round_half_up(fallback_img.height * sy))
                    fallback_scaled = fallback_img.resize((fw, fh),
                                                          Image.NEAREST)
                glyph_scaled = fallback_scaled
                tx_pixels = round_half_up(fallback_translate[0] *
                                          (2.0 * page_scale_x))
                ty_pixels = round_half_up(fallback_translate[1] *
                                          (2.0 * page_scale_y))

                # erase tile and paste fallback
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
                    continue
                paste_with_clip(out_im, glyph_scaled, dest_left, dest_top,
                                clip_left, clip_top, clip_right, clip_bottom)
                continue

            # Here: used_font is None and either uchar != U+0000 (Unifont glyph
            # remains) OR uchar==U+0000 but no fallback available. We only
            # place a width marker when the Unifont glyph remains (uchar not
            # U+0000) AND an override applies.
            if not is_json_null_entry(uchar):
                width_val = find_override_for_cp(width_overrides, cp)
                if width_val is not None:
                    # compute marker pixel position inside the tile:
                    # x = tile_left + round_half_up((width-1) * page_scale_x)
                    # y = tile_top + round_half_up((16-1) * page_scale_y)
                    x_offset = round_half_up((width_val - 1) * page_scale_x)
                    y_offset = round_half_up(bottom_index_0 * page_scale_y)
                    px = tile_left + x_offset
                    py = tile_top + y_offset
                    # check bounds and existing pixel (only write if fully
                    # transparent)
                    if (px >= tile_left and px < tile_left + target_tile_w and
                            py >= tile_top and py < tile_top + target_tile_h):
                        r, g, b, a = out_im.getpixel((px, py))
                        if a == 0:
                            # Use (0,0,0,1) => alpha 1/255 (very faint)
                            out_im.putpixel((px, py), (0, 0, 0, 1))
                            if verbose:
                                print(f"Marked width for U+{cp:04X} at "
                                      f"page {page_hex} tile ({row},{col}) -> "
                                      f"pixel ({px},{py}), width={width_val}")
                    else:
                        if verbose:
                            print(f"Override pixel for U+{cp:04X} falls outside "
                                  "tile bounds; skipping")
                # else: no override -> leave Unifont glyph as-is
            # else: uchar is null and no fallback -> leave tile empty

    if page_hex == "E0" or page_hex == "E1":
        if verbose:
            print(f"Skipping page {page_hex} (emoticons)")
        return
    out_path = os.path.join(out_dir, f"glyph_{page_hex}.png")
    os.makedirs(out_dir, exist_ok=True)
    write_png(out_im, out_path)
    if verbose:
        print(f"Wrote merged page: {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="merge_fonts_with_width_overrides.py",
        description="Merge pixel fonts into Unifont pages (with optional "
                    "unifont width overrides)"
    )
    p.add_argument("--config", default="merger_config.json",
                   help="Merger config JSON (default: merger_config.json)")
    p.add_argument("--unicode-dir", default="unicode",
                   help="Directory with unicode_XX.png/json (default: unicode)")
    p.add_argument("--pixel-dir", default="pixel",
                   help="Directory with pixel fonts and fallback.png "
                        "(default: pixel)")
    p.add_argument("--out-dir", default="out",
                   help="Output directory (default: out)")
    p.add_argument("--overrides", default="",
                   help=("Path to unifont_width_overrides.json "
                         "(default: unifont_width_overrides.json)"))
    p.add_argument("--verbose", action="store_true",
                   help="Verbose messages")
    args = p.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_json(args.config)
    cfg_fonts = cfg.get("fonts")
    if not isinstance(cfg_fonts, list):
        print("merger_config.json must contain a 'fonts' array",
              file=sys.stderr)
        return 2
    try:
        fallback_translate = validate_translate(
            cfg.get("fallback_translate", [0, 0]))
    except Exception as e:
        print(f"Invalid fallback_translate: {e}", file=sys.stderr)
        return 2

    pixel_fonts = load_pixel_fonts(cfg_fonts, args.pixel_dir,
                                   verbose=args.verbose)

    fallback_path = os.path.join(args.pixel_dir, "fallback.png")
    fallback_img = load_fallback(fallback_path)
    if fallback_img is None and args.verbose:
        print("Warning: fallback.png not found in pixel/; missing fallback "
              "glyphs will be left untouched", file=sys.stderr)

    try:
        if args.overrides:
            width_overrides = load_width_overrides(args.overrides)
            if args.verbose and width_overrides:
                print(f"Loaded {len(width_overrides)} width override ranges "
                      f"from {args.overrides}")
    except Exception as e:
        print(f"Error loading overrides: {e}", file=sys.stderr)
        return 2

    pages = list_unicode_pages(args.unicode_dir)
    if not pages:
        print("No unicode_XX pages found in", args.unicode_dir, file=sys.stderr)
        return 1

    for page_int, png_path, json_path in pages:
        try:
            merge_page(page_int, png_path, json_path, pixel_fonts,
                       fallback_img, fallback_translate, args.out_dir,
                       width_overrides, verbose=args.verbose)
        except Exception as e:
            print(f"Error processing page {page_int:02X}: {e}", file=sys.stderr)
            continue

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())