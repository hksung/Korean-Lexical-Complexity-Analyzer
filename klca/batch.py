from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .core import (
    CompositionalityIndices,
    DiversityIndices,
    FrequencyIndices,
    Normalize,
    SOAIndices,
    SophisticationIndices,
    VocabularyGradeTypeIndices,
    load_eojeol_sqlite_db,
    load_grade_level_rows,
    load_ko_stanza_local,
    load_morpheme_sqlite_db,
    dep_file_path,
    parameters,
)


@lru_cache(maxsize=1)
def load_soph_dbs() -> tuple[Any, Any]:
    mo_db = load_morpheme_sqlite_db()
    if mo_db is None:
        raise FileNotFoundError("Missing SQLite dependency file: dep_files/morpheme_db.sqlite")

    eo_db = load_eojeol_sqlite_db()
    if eo_db is None:
        raise FileNotFoundError("Missing SQLite dependency file: dep_files/eojeol_db.sqlite")

    return mo_db, eo_db


@lru_cache(maxsize=1)
def load_soa_dbs() -> tuple[dict, dict]:
    with open(dep_file_path("dep_files/soa_morph.json"), "r", encoding="utf-8") as f:
        mo_db = json.load(f)
    with open(dep_file_path("dep_files/soa_eojeol.json"), "r", encoding="utf-8") as f:
        eo_db = json.load(f)
    return mo_db, eo_db


@lru_cache(maxsize=1)
def load_grade_rows() -> list[dict[str, Any]]:
    return load_grade_level_rows()


def ensure_nlp(use_gpu: bool | None = None, force_reload: bool = False) -> None:
    if use_gpu is not None:
        parameters.use_gpu = use_gpu

    if parameters.nlp is None or force_reload:
        parameters.unit = "morpheme"
        parameters.sync()
        parameters.nlp = load_ko_stanza_local(
            model_dir=None,
            use_gpu=parameters.use_gpu,
            processors=parameters.processors_morpheme,
        )


def _build_norms(text: str) -> tuple[Normalize, Normalize]:
    parameters.unit = "morpheme"
    parameters.sync()
    norm_mo = Normalize(text=text, params=parameters)

    parameters.unit = "eojeol"
    parameters.sync()
    norm_eo = Normalize(text=text, params=parameters)

    # Keep morpheme mode for indices that require POS/XPOS.
    parameters.unit = "morpheme"
    parameters.sync()
    return norm_mo, norm_eo


def compute_indices(text: str, debug: bool = False, ensure_pipeline: bool = True) -> Dict[str, Any]:
    if ensure_pipeline:
        ensure_nlp(force_reload=False)

    norm_mo, norm_eo = _build_norms(text)
    soph_mo_db, soph_eo_db = load_soph_dbs()
    soa_mo_db, soa_eo_db = load_soa_dbs()
    grade_rows = load_grade_rows()

    freq = FrequencyIndices(morpheme_norm=norm_mo, eojeol_norm=norm_eo)
    div = DiversityIndices(morpheme_norm=norm_mo, eojeol_norm=norm_eo)
    soph = SophisticationIndices(
        morpheme_norm=norm_mo,
        eojeol_norm=norm_eo,
        mo_db=soph_mo_db,
        eo_db=soph_eo_db,
        debug=debug,
    )
    soa = SOAIndices(
        morpheme_norm=norm_mo,
        eojeol_norm=norm_eo,
        mo_db=soa_mo_db,
        eo_db=soa_eo_db,
        debug=debug,
    )
    comp = CompositionalityIndices(text=text, params=parameters, debug=debug)
    grade = VocabularyGradeTypeIndices(text=text, params=parameters, lex_db_rows=grade_rows)

    out: Dict[str, Any] = {}
    out.update(freq.vald)
    out.update(div.vald)
    out.update(soph.vald)
    out.update(soa.vald)
    out.update(comp.vald)
    out.update(grade.vald)
    #out["n_chars"] = len(text)
    return out


def _iter_text_files(input_dir: Path, recursive: bool, suffixes: Sequence[str]) -> Iterable[Path]:
    normalized = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes}
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in normalized]
    for p in sorted(files):
        yield p


def process_file(path: str | Path, encoding: str = "utf-8", debug: bool = False) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding=encoding, errors="ignore")
    rec = compute_indices(text=text, debug=debug, ensure_pipeline=True)
    rec["filename"] = p.name
    return rec


def process_folder(
    input_dir: str | Path,
    recursive: bool = False,
    suffixes: Sequence[str] = (".txt",),
    encoding: str = "utf-8",
    debug: bool = False,
    use_gpu: bool | None = None,
    force_reload_nlp: bool = True,
) -> List[Dict[str, Any]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    ensure_nlp(use_gpu=use_gpu, force_reload=force_reload_nlp)
    files = list(_iter_text_files(root, recursive=recursive, suffixes=suffixes))
    total = len(files)
    if total == 0:
        print(f"No matching files found in: {root}", flush=True)
        return []

    rows: List[Dict[str, Any]] = []
    for idx, p in enumerate(files, start=1):
        print(f"[{idx}/{total}] Processing: {p}", flush=True)
        text = p.read_text(encoding=encoding, errors="ignore")
        row = compute_indices(text=text, debug=debug, ensure_pipeline=False)
        row["filename"] = p.name
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        outp.write_text("", encoding="utf-8")
        return

    keys = set()
    for r in rows:
        keys.update(r.keys())
    preferred = ["filename"]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    with outp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_rows(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    outp = Path(output_path)
    if outp.suffix.lower() == ".jsonl":
        write_jsonl(rows, outp)
    else:
        write_csv(rows, outp)
