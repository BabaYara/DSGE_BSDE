#!/usr/bin/env python3
"""
Fix common LaTeX macro typos across .tex files:
- extbf{...}  -> \textbf{...}
- extit{...}  -> \textit{...}
- ext{...}    -> \text{...}
Also fixes the backslashed variants (\extbf, \extit, \ext) to the canonical forms
and optionally removes temporary shim definitions (\providecommand{\extbf}...).

Safety features:
- Skips verbatim/code environments: verbatim, Verbatim, lstlisting, minted.
- Skips lines that are fully commented out.
- Supports dry-run with unified diff output.

Usage (PowerShell examples):
  python scripts/fix_tex_macros.py Tex/BSDE_31.tex --write --backup
  python scripts/fix_tex_macros.py Tex/*.tex --dry-run

Exit code is nonzero if changes were suggested in --dry-run mode.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import io
import os
import re
import sys
from typing import Iterable, List, Tuple


VERBATIM_ENVS = {"verbatim", "Verbatim", "lstlisting", "minted"}


def _apply_replacements(text: str) -> str:
    """Apply macro replacements outside code-like environments and comments."""
    out_lines: List[str] = []
    in_code = False
    code_env: str | None = None

    # Patterns: both stray (no backslash) and backslashed variants
    patterns: List[Tuple[re.Pattern[str], str]] = [
        # extbf -> \textbf
        (re.compile(r"(?<!\\)\bextbf\s*\{"), r"\\textbf{"),
        (re.compile(r"\\extbf\s*\{"), r"\\textbf{"),
        # extit -> \textit
        (re.compile(r"(?<!\\)\bextit\s*\{"), r"\\textit{"),
        (re.compile(r"\\extit\s*\{"), r"\\textit{"),
        # ext{ -> \text{
        (re.compile(r"(?<!\\)\bext\s*\{"), r"\\text{"),
        (re.compile(r"\\ext\s*\{"), r"\\text{"),
    ]

    begin_env_re = re.compile(r"\\begin\{([^}]+)\}")
    end_env_re = re.compile(r"\\end\{([^}]+)\}")

    for raw_line in text.splitlines(keepends=False):
        line = raw_line

        # Track begin/end of code-like environments
        if not in_code:
            m = begin_env_re.search(line)
            if m and m.group(1) in VERBATIM_ENVS:
                in_code = True
                code_env = m.group(1)
        else:
            m = end_env_re.search(line)
            if m and code_env and m.group(1) == code_env:
                in_code = False
                code_env = None

        # Skip replacements inside code-like environments
        if in_code:
            out_lines.append(line)
            continue

        # Skip fully commented lines
        stripped = line.lstrip()
        if stripped.startswith("%"):
            out_lines.append(line)
            continue

        # Perform replacements
        new_line = line
        for pat, repl in patterns:
            new_line = pat.sub(repl, new_line)

        out_lines.append(new_line)

    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def _remove_shims(text: str) -> str:
    """Remove providecommand shims for \extbf, \extit, and \ext if they exist.
    Only remove when exact shim pattern is found to avoid accidental deletions.
    """
    lines = text.splitlines(keepends=False)
    keep: List[str] = []

    shim_res = [
        re.compile(r"^\\providecommand\{\\extbf\}\[1\]\{\\textbf\{#1\}\}\s*$"),
        re.compile(r"^\\providecommand\{\\extit\}\[1\]\{\\textit\{#1\}\}\s*$"),
        re.compile(r"^\\providecommand\{\\ext\}\[1\]\{\\text\{#1\}\}\s*$"),
    ]

    for line in lines:
        if any(pat.match(line.strip()) for pat in shim_res):
            # drop this line
            continue
        keep.append(line)

    return "\n".join(keep) + ("\n" if (len(lines) > 0 and text.endswith("\n")) else "")


def process_file(path: str, write: bool, backup: bool, remove_shims: bool) -> Tuple[bool, str]:
    with io.open(path, "r", encoding="utf-8") as f:
        original = f.read()

    updated = _apply_replacements(original)
    if remove_shims:
        updated = _remove_shims(updated)

    changed = (updated != original)
    if changed and write:
        if backup:
            bak = path + ".bak"
            with io.open(bak, "w", encoding="utf-8") as fb:
                fb.write(original)
        with io.open(path, "w", encoding="utf-8") as fw:
            fw.write(updated)

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=path,
            tofile=path + (" (fixed)" if changed else " (no changes)"),
            n=3,
        )
    )
    return changed, diff


def find_files(patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.tex"), recursive=True))
        else:
            files.extend(glob.glob(p, recursive=True))
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fix LaTeX macro typos (extbf/extit/ext{...}).")
    ap.add_argument("paths", nargs="+", help="Files, directories, or glob patterns to process.")
    ap.add_argument("--dry-run", action="store_true", help="Show diffs without writing changes.")
    ap.add_argument("--write", action="store_true", help="Write changes in-place.")
    ap.add_argument("--backup", action="store_true", help="Create .bak backups when writing.")
    ap.add_argument("--remove-shims", action="store_true", help="Remove temporary providecommand shims for \\extbf/\\extit/\\ext.")

    args = ap.parse_args(argv)
    if not args.dry_run and not args.write:
        # default to dry-run for safety
        args.dry_run = True

    files = find_files(args.paths)
    if not files:
        print("No files matched.")
        return 0

    any_changes = False
    for fp in files:
        changed, diff = process_file(fp, write=args.write, backup=args.backup, remove_shims=args.remove_shims)
        if args.dry_run or not args.write:
            if diff:
                sys.stdout.write(diff)
        any_changes = any_changes or changed

    if args.dry_run:
        return 1 if any_changes else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
