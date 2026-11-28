#!/usr/bin/env python3
"""Build every model in `models.json`: data, features, training, and evaluation.

Run from the repository root. `WORK` is any directory outside the
repository; it receives the corpus, the features, and the trained models.

    python3 -B tools/train/build_models.py --work "$WORK"
    python3 -B tools/train/build_models.py --work "$WORK" --only compact --install

Steps, each skipped when its output exists unless `--force` is given:

1. Build the `fingerprint_corpus` and `score` examples in release mode.
2. Generate each data split with `dataset.py`, in parallel.
3. Extract fingerprints with the library through `fingerprint_corpus`.
4. Train each model with `train.py`.
5. Score the hand-written development set in `tests/data/` with each model
   through the `score` example, so that the result includes the windowing
   of the library.

`--install` copies each model and its golden vectors to `models/`. The
summary of every model is written to `$WORK/summary.json` and printed as a
Markdown table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CATEGORIES = (
    "malformed", "obfuscation", "sql_injection", "markup_injection",
    "path_traversal", "command_injection", "template_injection", "header_injection",
)
THRESHOLD = 128


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("+ " + " ".join(str(part) for part in command))
    return subprocess.run(command, check=True, **kwargs)


def build_examples() -> Path:
    run(["cargo", "build", "--release", "--locked", "--example", "fingerprint_corpus", "--example", "score"], cwd=ROOT)
    return ROOT / "target" / "release" / "examples"


def make_split(work: Path, name: str, split: dict, join_rate: float, examples: Path, force: bool) -> None:
    corpus = work / f"{name}.bin"
    if force or not corpus.exists():
        run([sys.executable, "-B", str(HERE / "dataset.py"), "--seed", str(split["seed"]), "--count", str(split["count"]),
             "--join-rate", str(join_rate), "--out", str(corpus)])
    if force or not (work / f"{name}.x").exists():
        run([str(examples / "fingerprint_corpus"), str(corpus), str(work / f"{name}.x"), str(work / f"{name}.y")])


def train_model(work: Path, model: dict, force: bool) -> Path:
    out = work / "out"
    path = out / f"{model['name']}.s2fp"
    if force or not path.exists():
        command = [sys.executable, "-B", str(HERE / "train.py"), "--data", str(work), "--out", str(out),
                   "--name", model["name"], "--hidden", *[str(h) for h in model["hidden"]],
                   "--epochs", str(model["epochs"]), "--quantized-epochs", str(model["quantized_epochs"]),
                   "--seed", str(model["seed"])]
        for key in ("benign_weight", "learning_rate", "weight_decay", "input_dropout", "batch_size"):
            if key in model:
                command += ["--" + key.replace("_", "-"), str(model[key])]
        run(command, stdout=subprocess.DEVNULL)
    return path


def score_lines(examples: Path, model: Path, lines: list[bytes]) -> list[list[int]]:
    result = run([str(examples / "score"), str(model)], input=b"\n".join(lines) + b"\n", capture_output=True)
    rows = [line for line in result.stdout.decode("ascii").splitlines() if not line.startswith("#")]
    if len(rows) != len(lines):
        raise RuntimeError("the score example returned an unexpected number of rows")
    return [[int(value) for value in row.split("\t")[: len(CATEGORIES)]] for row in rows]


def development_set(examples: Path, model: Path) -> dict:
    attacks = []
    for line in (ROOT / "tests" / "data" / "attacks.tsv").read_bytes().splitlines():
        category, payload = line.split(b"\t", 1)
        attacks.append((CATEGORIES.index(category.decode()), payload))
    benign = (ROOT / "tests" / "data" / "benign.txt").read_bytes().splitlines()
    attack_scores = score_lines(examples, model, [payload for _, payload in attacks])
    benign_scores = score_lines(examples, model, benign)
    missed = [index + 1 for index, ((category, _), scores) in enumerate(zip(attacks, attack_scores))
              if scores[category] < THRESHOLD]
    flagged = [index + 1 for index, scores in enumerate(benign_scores) if max(scores) >= THRESHOLD]
    return {
        "attacks": len(attacks), "attacks_flagged": len(attacks) - len(missed), "attack_lines_missed": missed,
        "benign": len(benign), "benign_flagged": len(flagged), "benign_lines_flagged": flagged,
    }


def summarize(model: dict, metrics: dict, development: dict, size: int) -> dict:
    test = metrics["test"]
    return {
        "name": model["name"], "dimensions": metrics["dimensions"], "bytes": size,
        "benign_any_fpr@128": test["benign_any_fpr@128"], "attack_any_recall@128": test["attack_any_recall@128"],
        "benign_any_fpr@192": test["benign_any_fpr@192"], "attack_any_recall@192": test["attack_any_recall@192"],
        "mean_auc": round(sum(entry["auc"] for entry in test["categories"].values()) / len(CATEGORIES), 4),
        "quantization_drift": metrics["quantization_drift"]["mean_abs_score"],
        "development": development,
    }


def markdown(rows: list[dict]) -> str:
    out = ["| Model | Layers | Bytes | Mean AUC | Attacks flagged at 128 | Benign flagged at 128 | "
           "Development attacks | Development benign |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        dev = row["development"]
        out.append(
            f"| `{row['name']}` | {', '.join(str(d) for d in row['dimensions'])} | {row['bytes']:,} | {row['mean_auc']:.3f} | "
            f"{100 * row['attack_any_recall@128']:.1f}% | {100 * row['benign_any_fpr@128']:.2f}% | "
            f"{dev['attacks_flagged']} of {dev['attacks']} | {dev['benign_flagged']} of {dev['benign']} |"
        )
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="directory outside the repository for data and models")
    parser.add_argument("--config", type=Path, default=HERE / "models.json")
    parser.add_argument("--only", nargs="*", help="names of the models to build; all by default")
    parser.add_argument("--install", action="store_true", help="copy the models and golden vectors to models/")
    parser.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    arguments = parser.parse_args(argv)

    config = json.loads(arguments.config.read_text())
    models = [m for m in config["models"] if not arguments.only or m["name"] in arguments.only]
    if arguments.only and len(models) != len(arguments.only):
        parser.error("unknown model name in --only")
    work = arguments.work.resolve()
    if work == ROOT or ROOT in work.parents:
        parser.error("--work must be outside the repository")
    work.mkdir(parents=True, exist_ok=True)

    examples = build_examples()
    data = config["data"]
    with ThreadPoolExecutor(max_workers=len(data["splits"])) as pool:
        jobs = [pool.submit(make_split, work, name, split, data["join_rate"], examples, arguments.force)
                for name, split in data["splits"].items()]
        for job in jobs:
            job.result()

    rows = []
    for model in models:
        path = train_model(work, model, arguments.force)
        metrics = json.loads(path.with_suffix(".json").read_text())
        development = development_set(examples, path)
        rows.append(summarize(model, metrics, development, path.stat().st_size))
        if arguments.install:
            shutil.copyfile(path, ROOT / "models" / f"{model['name']}.s2fp")
            shutil.copyfile(path.with_suffix(".golden"), ROOT / "models" / f"{model['name']}.golden")
            log(f"installed models/{model['name']}.s2fp")
    (work / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
