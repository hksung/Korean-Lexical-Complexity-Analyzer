from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import process_file, process_folder, write_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Korean text indices for one file or a folder of text files."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_file = sub.add_parser("file", help="Process a single text file.")
    p_file.add_argument("--input-file", required=True, help="Path to a text file.")
    p_file.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8).")
    p_file.add_argument("--debug", action="store_true", help="Print debug info from index classes.")
    p_file.add_argument(
        "--output",
        default="",
        help="Optional output path (.json/.jsonl/.csv). If omitted, print JSON to stdout.",
    )

    p_folder = sub.add_parser("folder", help="Process all text files in a folder.")
    p_folder.add_argument("--input-dir", required=True, help="Directory containing text files.")
    p_folder.add_argument("--recursive", action="store_true", help="Recursively scan subfolders.")
    p_folder.add_argument(
        "--suffixes",
        nargs="+",
        default=[".txt"],
        help="File suffixes to include (default: .txt).",
    )
    p_folder.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8).")
    p_folder.add_argument("--debug", action="store_true", help="Print debug info from index classes.")
    p_folder.add_argument("--use-gpu", action="store_true", help="Enable GPU for Stanza if available.")
    p_folder.add_argument(
        "--output",
        default="outputs/results.csv",
        help="Output path (.csv or .jsonl). Default: outputs/results.csv",
    )
    return parser


def _write_single_record(record: dict, output_path: str) -> None:
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ext = outp.suffix.lower()

    if ext == ".json":
        outp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if ext == ".jsonl":
        outp.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    # Default to CSV one-row output.
    write_rows([record], outp)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "file":
        print(f"Processing: {args.input_file}", flush=True)
        record = process_file(args.input_file, encoding=args.encoding, debug=args.debug)
        if args.output:
            _write_single_record(record, args.output)
            print(f"Wrote: {args.output}")
        else:
            print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    print(f"Scanning folder: {args.input_dir}", flush=True)
    rows = process_folder(
        input_dir=args.input_dir,
        recursive=args.recursive,
        suffixes=args.suffixes,
        encoding=args.encoding,
        debug=args.debug,
        use_gpu=args.use_gpu,
    )
    write_rows(rows, args.output)
    print(f"Processed files: {len(rows)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

