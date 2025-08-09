#!/usr/bin/env python3
"""
filter_symbols_one.py

Filter a single char-mapping JSON by allowed symbols.

Usage:
  python filter_symbols_one.py [--symbols SYMBOL_FILE] MAPPING_JSON
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Set, Tuple

HEX_U_RE = re.compile(r'^[\\/][uU]\+?([0-9A-Fa-f]{1,6})$')
U_RE = re.compile(r'^[uU]\+?([0-9A-Fa-f]{1,6})$')
OX_RE = re.compile(r'^0[xX]([0-9A-Fa-f]{1,6})$')
HEX_ONLY_RE = re.compile(r'^[0-9A-Fa-f]{1,6}$')


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=True, indent=2)
        fh.write('\n')


def token_to_chars(token: str) -> List[str]:
    """
    Convert tokens like '/u1234', '\\u1234', 'U+1234', '0x1234', '1234',
    or literal characters into actual character strings.
    """
    t = token.strip()
    if not t:
        return []
    m = HEX_U_RE.match(t)
    if m:
        cp = int(m.group(1), 16)
        if cp <= 0x10FFFF:
            return [chr(cp)]
        return []
    m = U_RE.match(t)
    if m:
        cp = int(m.group(1), 16)
        if cp <= 0x10FFFF:
            return [chr(cp)]
        return []
    m = OX_RE.match(t)
    if m:
        cp = int(m.group(1), 16)
        if cp <= 0x10FFFF:
            return [chr(cp)]
        return []
    m = HEX_ONLY_RE.match(t)
    if m and len(t) > 1:
        # treat multi-digit hex-only token as a codepoint
        cp = int(m.group(1), 16)
        if cp <= 0x10FFFF:
            return [chr(cp)]
        return []
    if len(t) == 1:
        return [t]
    # attempt to decode any embedded \u escapes
    if '\\u' in t:
        try:
            dec = t.encode('utf-8').decode('unicode_escape')
            return list(dec)
        except Exception:
            pass
    return list(t)


def parse_symbols_file(path: str) -> Set[str]:
    obj = load_json(path)
    symbols = obj.get('symbols')
    if not isinstance(symbols, list):
        raise ValueError(f"{path}: missing 'symbols' array")
    allowed: Set[str] = set()
    for item in symbols:
        if not isinstance(item, str):
            continue
        # split tokens by whitespace, comma, semicolon, pipe
        tokens = re.split(r'[\s,;|]+', item)
        for tok in tokens:
            if not tok:
                continue
            for ch in token_to_chars(tok):
                allowed.add(ch)
    return allowed


def process_mapping_file(path: str, allowed: Set[str],
                         out_path: str) -> None:
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: top-level JSON object required")
    chars = obj.get('chars')
    if not isinstance(chars, list):
        raise ValueError(f"{path}: missing 'chars' array")
    new_rows: List[str] = []
    for row in chars:
        row_s = str(row)
        new_chars: List[str] = []
        for ch in row_s:
            if ch in allowed:
                new_chars.append(ch)
            else:
                new_chars.append('\x00')
        new_rows.append(''.join(new_chars))
    obj['chars'] = new_rows
    write_json(out_path, obj)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="filter_symbols_one.py")
    p.add_argument('--symbols', default='symbol_chars.json',
                   help='symbol_chars.json path (default symbol_chars.json)')
    p.add_argument('mapping', help='Single mapping JSON to filter')
    p.add_argument('--out', default=None,
                   help=('Optional output path. Default: '
                         '<mapping_basename>_symbols.json'))
    args = p.parse_args(argv)

    if not os.path.isfile(args.symbols):
        print(f"Symbol file not found: {args.symbols}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.mapping):
        print(f"Mapping file not found: {args.mapping}", file=sys.stderr)
        return 2

    try:
        allowed = parse_symbols_file(args.symbols)
    except Exception as e:
        print(f"Failed to parse symbols file: {e}", file=sys.stderr)
        return 2

    mapping_path = args.mapping
    if args.out:
        out_path = args.out
    else:
        dirn, base = os.path.split(mapping_path)
        name, _ext = os.path.splitext(base)
        out_path = os.path.join(dirn or '.', f"{name}_symbols.json")

    try:
        process_mapping_file(mapping_path, allowed, out_path)
    except Exception as e:
        print(f"Error processing mapping file: {e}", file=sys.stderr)
        return 1

    print(f"Wrote filtered mapping: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())