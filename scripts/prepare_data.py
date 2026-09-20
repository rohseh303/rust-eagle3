"""Build a pinned Rust pilot; never use benchmark solutions as training targets."""
import argparse
import ast
import collections
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value):
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_mapping(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)
    if not isinstance(result, dict):
        raise ValueError("Expected a mapping")
    return result


def split_for(crate):
    bucket = int(digest("rust-draft-v1:" + crate)[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "heldout"


def strand_prompt(row):
    data = parse_mapping(row["input_data"])
    if row["task_category"] == "code_completion":
        if not data.get("prefix") or "suffix" not in data:
            raise ValueError("Missing completion context")
        return (
            "Fill the missing region in this Rust code. Return only the missing code, "
            "without Markdown fences or explanation.\n\nPREFIX:\n" + data["prefix"]
            + "\n\nSUFFIX:\n" + data["suffix"]
        )
    if row["task_category"] == "code_generation":
        if not data.get("description"):
            raise ValueError("Missing specification")
        return (
            "Implement the requested Rust function. Return the complete function and any "
            "necessary helper functions, without Markdown fences or explanation.\n\n"
            + json.dumps(data, ensure_ascii=False, indent=2)
        )
    raise ValueError("Not a generation task")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def download(cache, name, source, filename):
    path = cache / name
    if not path.exists():
        tmp = path.with_suffix(".partial")
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{filename}"
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--train-size", type=int, default=512)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "configs/sources.json").read_text())
    source = sources["training_data"]
    train_file = download(args.cache, "strand-train.parquet", source, source["file"])
    benchmark = sources["evaluation_data"]
    benchmark_rows = {}
    source_files = [train_file]
    for language in ("rust", "python"):
        path = download(args.cache, f"humaneval-{language}.parquet", benchmark,
                        f"{language}/test-00000-of-00001.parquet")
        source_files.append(path)
        benchmark_rows[language] = pq.read_table(path).to_pylist()

    # Remove exact benchmark spec/name matches. This does not prove semantic
    # decontamination or remove contamination inherited by the frozen target.
    forbidden_names = {r["entry_point"] for rows in benchmark_rows.values() for r in rows}
    forbidden_specs = {normalize(r["docstring"]) for rows in benchmark_rows.values() for r in rows}
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    seen = set()
    for row in pq.read_table(train_file).to_pylist():
        if row["task_category"] not in {"code_generation", "code_completion"}:
            continue
        crate = row.get("crate_name", "").strip()
        if not crate:
            counts["missing_crate"] += 1
            continue
        try:
            prompt = strand_prompt(row)
            reference = parse_mapping(row["output_data"])
            code = reference.get("completion", reference.get("code", ""))
            # Deliberately select longer source tasks for this decode-heavy pilot.
            if not isinstance(code, str) or len(code) < 400 or len(prompt) > 12000:
                counts["length_filter"] += 1
                continue
        except (ValueError, SyntaxError, TypeError):
            counts["malformed"] += 1
            continue
        normalized = normalize(prompt)
        if any(re.search(r"\b" + re.escape(name) + r"\b", prompt) for name in forbidden_names) or any(
            len(spec) > 60 and spec in normalized for spec in forbidden_specs
        ):
            counts["benchmark_overlap"] += 1
            continue
        key = digest(normalized)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        groups[split_for(crate)].append({
            "id": "strand-" + key[:20], "crate": crate, "task": row["task_category"],
            "prompt": prompt, "source": source["repo"], "revision": source["revision"],
            "source_reference_chars": len(code),
        })
    rng = random.Random(sources["seed"])
    selected = {}
    for split, limit in (("train", args.train_size), ("validation", 64), ("heldout", 128)):
        rng.shuffle(groups[split])
        rows = groups[split][:limit]
        if len(rows) != limit:
            raise RuntimeError(f"Insufficient {split}: {len(rows)}/{limit}")
        selected[split] = rows
        write_jsonl(args.output / f"{split}.jsonl", rows)
    for language, rows in benchmark_rows.items():
        write_jsonl(args.output / f"humaneval-{language}.jsonl", [{
            "id": r["task_id"], "task": f"humaneval-{language}",
            "prompt": r["instruction"] + "\nReturn only the complete function without Markdown fences.",
            "source": benchmark["repo"], "revision": benchmark["revision"],
        } for r in rows])
        # References stay in a separate file, never supplied to target regeneration.
        write_jsonl(args.output / "references" / f"humaneval-{language}.jsonl", rows)
    for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        assert not ({r["crate"] for r in selected[a]} & {r["crate"] for r in selected[b]})
    manifest = {
        "sources": sources, "filter_counts": dict(counts),
        "eligible_counts": {s: len(v) for s, v in groups.items()},
        "selected_counts": {s: len(v) for s, v in selected.items()},
        "selected_crates": {s: len({r['crate'] for r in v}) for s, v in selected.items()},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.jsonl')},
        "status": "Prompts only; target responses have not been generated.",
        "limitations": ["Synthetic source data; provider quality claims not independently reproduced.",
                       "Crate-disjoint splits; exact dedup and benchmark spec/name exclusions only.",
                       "No guarantee against semantic or target pretraining contamination.",
                       "Long-output selection uses source answer length, not observed target output length."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:manifest[k] for k in ['selected_counts','selected_crates','eligible_counts']}, indent=2))


if __name__ == "__main__":
    main()
