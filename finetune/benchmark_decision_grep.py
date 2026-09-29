#!/usr/bin/env python3
"""Benchmark ranked path retrieval against JSONL judgments."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


K = 5
METRICS = (
    "recall_at_5",
    "precision_at_5_lower_bound",
    "judged_precision_at_5",
    "judged_coverage_at_5",
    "ndcg_at_5",
    "mrr",
)
WINNER_METRICS = METRICS + ("latency_ms",)


def normalize_path(value):
    """Normalize path separators and case for matching."""
    text = str(value).replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    while parts and parts[0].lower() == ".":
        parts.pop(0)
    return "/".join(parts).lower()


def read_jsonl(filename):
    rows = []
    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{filename}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def path_set(values):
    return {normalize_path(value) for value in values}


def weak_label_reasons(judgments):
    reasons = []
    for row in judgments:
        relevant = [normalize_path(value) for value in row.get("relevant_paths", [])]
        nonrelevant = [normalize_path(value) for value in row.get("nonrelevant_paths", [])]
        if len(relevant) != len(set(relevant)) or len(nonrelevant) != len(set(nonrelevant)):
            reasons.append(f"duplicate labels for query_id={row.get('query_id')}")
        overlap = set(relevant) & set(nonrelevant)
        if overlap:
            reasons.append(f"conflicting labels for query_id={row.get('query_id')}")
        if not relevant and not nonrelevant:
            reasons.append(f"empty labels for query_id={row.get('query_id')}")
        if row.get("weak_labels") is True or row.get("weak_label") is True:
            reasons.append(f"weakly supervised labels for query_id={row.get('query_id')}")
    return reasons


def query_metrics(judgment, run):
    relevant = path_set(judgment.get("relevant_paths", []))
    nonrelevant = path_set(judgment.get("nonrelevant_paths", []))
    retrieved = [normalize_path(value) for value in run.get("retrieved_paths", [])][:K]
    labels = [path in relevant or path in nonrelevant for path in retrieved]
    hits = [path in relevant for path in retrieved]
    relevant_hits = sum(hits)
    judged_hits = sum(labels)
    recall = relevant_hits / len(relevant) if relevant else None
    judged_precision = relevant_hits / judged_hits if judged_hits else None
    coverage = judged_hits / len(retrieved) if retrieved else 0.0
    dcg = sum((1.0 / math.log2(index + 2)) for index, hit in enumerate(hits) if hit)
    ideal_count = min(K, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    ndcg = dcg / ideal_dcg if ideal_dcg else None
    first_hit = next((index for index, hit in enumerate(hits, 1) if hit), None)
    return {
        "recall_at_5": recall,
        "precision_at_5_lower_bound": relevant_hits / K,
        "judged_precision_at_5": judged_precision,
        "judged_coverage_at_5": coverage,
        "returned_at_5": len(retrieved),
        "ndcg_at_5": ndcg,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "latency_ms": run.get("latency_ms"),
    }


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def percentile(values, percentile_value):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def winner_names(rows, metric, lower_is_better=False):
    usable = [(row["system"], row["metrics"].get(metric)) for row in rows]
    usable = [(name, value) for name, value in usable if value is not None]
    if not usable:
        return []
    best = (min if lower_is_better else max)(value for _, value in usable)
    return sorted(name for name, value in usable if value == best)


def benchmark(judgment_rows, run_rows):
    judgments = {str(row["query_id"]): row for row in judgment_rows}
    weak = weak_label_reasons(judgment_rows)
    per_query = {}
    by_system = {}
    for run in run_rows:
        query_id = str(run["query_id"])
        if query_id not in judgments:
            raise ValueError(f"run query_id has no judgment: {query_id}")
        system = str(run["system"])
        entry = {
            "system": system,
            "query_id": query_id,
            "metrics": query_metrics(judgments[query_id], run),
        }
        per_query.setdefault(query_id, []).append(entry)
        by_system.setdefault(system, []).append(entry)

    paired = {}
    for query_id, rows in sorted(per_query.items()):
        paired[query_id] = {
            metric: winner_names(rows, metric, lower_is_better=metric == "latency_ms")
            for metric in WINNER_METRICS
        }

    aggregate = {}
    for system, rows in sorted(by_system.items()):
        aggregate[system] = {
            metric: mean([row["metrics"].get(metric) for row in rows])
            for metric in METRICS
        }
        aggregate[system]["p95_query_latency_ms"] = percentile(
            [row["metrics"].get("latency_ms") for row in rows], 95
        )

    coverage_values = [
        row["metrics"]["judged_coverage_at_5"]
        for rows in per_query.values()
        for row in rows
    ]
    warnings = []
    if any(value < 1.0 for value in coverage_values):
        warnings.append("top-5 judgment coverage is below 100% for at least one run")
    if any(
        row["metrics"]["returned_at_5"] < K
        for rows in per_query.values()
        for row in rows
    ):
        warnings.append("one or more systems returned fewer than five files; precision@5 is normalized by five")
    if weak:
        warnings.append("one or more judgment labels are weak")

    return {
        "query_count": len(judgments),
        "systems": sorted(by_system),
        "aggregate": aggregate,
        "per_query": {
            query_id: sorted(rows, key=lambda row: row["system"])
            for query_id, rows in sorted(per_query.items())
        },
        "paired_winners": paired,
        "warnings": warnings,
        "weak_label_reasons": weak,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark ranked path retrieval from judgment and run JSONL files."
    )
    parser.add_argument("judgments", help="judgment JSONL file")
    parser.add_argument("runs", nargs="+", help="one or more run JSONL files")
    parser.add_argument("--out", metavar="FILE", help="also write JSON output to FILE")
    args = parser.parse_args(argv)
    try:
        run_rows = [row for filename in args.runs for row in read_jsonl(filename)]
        result = benchmark(read_jsonl(args.judgments), run_rows)
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        sys.stdout.write(output)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
