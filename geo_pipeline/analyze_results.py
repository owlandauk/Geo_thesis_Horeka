"""Offline diagnostics for geo_pipeline result JSON files.

Usage:
  python3 geo_pipeline/analyze_results.py --pred geo_pipeline/results/full_v5.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, median

from country_aliases import canonicalize_country, continent_of
from config import EVAL_THRESHOLDS


_THRESHOLD_LABELS = {
    1: "Street <1km",
    25: "City <25km",
    200: "Region <200km",
    750: "Country <750km",
    2500: "Continent <2500km",
}


def _top_mass(record: dict) -> float | None:
    posterior = record.get("country_posterior") or {}
    if not posterior:
        return None
    return max(float(v) for v in posterior.values())


def _country_conflicts(record: dict) -> list[str]:
    pred_country = canonicalize_country(record.get("pred_country") or "")
    if not pred_country:
        return []

    conflicts = []
    for field in ("pred_city", "pred_street"):
        child_country = canonicalize_country(record.get(field) or "")
        if child_country and child_country != pred_country:
            conflicts.append(field)
    return conflicts


def _gt_continent_from_coords(lat: float | None, lon: float | None) -> str | None:
    """Approximate continent from GPS coordinates for offline error slicing."""
    if lat is None or lon is None:
        return None

    if -170 <= lon <= -52 and 5 <= lat <= 83:
        return "North America"
    if -82 <= lon <= -34 and -56 <= lat <= 13:
        return "South America"
    if -25 <= lon <= 45 and 35 <= lat <= 72:
        return "Europe"
    if -20 <= lon <= 55 and -35 <= lat <= 38:
        return "Africa"
    if 110 <= lon <= 180 and -50 <= lat <= 5:
        return "Oceania"
    if -180 <= lon <= -120 and -30 <= lat <= 30:
        return "Oceania"
    if 25 <= lon <= 180 and -10 <= lat <= 80:
        return "Asia"
    return None


def _record_gt_continent(record: dict) -> str | None:
    return record.get("gt_continent") or _gt_continent_from_coords(
        record.get("gt_lat"), record.get("gt_lon")
    )


def _accuracy_by_threshold(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {str(thr): 0.0 for thr in EVAL_THRESHOLDS}
    return {
        str(thr): round(
            100.0
            * sum(float(r.get("dist_km", float("inf"))) <= thr for r in records)
            / n,
            2,
        )
        for thr in EVAL_THRESHOLDS
    }


def _bucket_accuracy(records: list[dict], key_fn) -> dict:
    buckets = defaultdict(list)
    for record in records:
        buckets[key_fn(record)].append(record)
    return {
        str(key): {"n": len(items), "accuracy": _accuracy_by_threshold(items)}
        for key, items in sorted(buckets.items(), key=lambda kv: str(kv[0]))
    }


def _mass_bucket(record: dict) -> str:
    mass = _top_mass(record)
    if mass is None:
        return "missing"
    if mass < 0.45:
        return "<0.45"
    if mass < 0.55:
        return "0.45-0.55"
    if mass < 0.65:
        return "0.55-0.65"
    return ">=0.65"


def analyze(records: list[dict]) -> dict:
    total = len(records)
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    for record in records:
        dist = float(record.get("dist_km", float("inf")))
        for thr in EVAL_THRESHOLDS:
            if dist <= thr:
                correct[thr] += 1

    unknown = sum(1 for r in records if not canonicalize_country(r.get("pred_country") or ""))
    masses = [m for r in records if (m := _top_mass(r)) is not None]
    source_counts = Counter(r.get("geocode_source") or "missing" for r in records)
    consistency_counts = Counter(r.get("country_consistency") or "missing" for r in records)
    conflicts = [r for r in records if _country_conflicts(r)]
    country_replaced = sum(1 for r in records if r.get("country_replaced"))
    country_web_enhanced = sum(1 for r in records if r.get("country_web_enhanced"))
    visual_deltas = [
        float(r["country_visual_delta"]) for r in records
        if r.get("country_visual_delta") is not None
    ]
    web_deltas = [
        float(r["country_web_delta"]) for r in records
        if r.get("country_web_delta") is not None
    ]
    country_stable_known = [r for r in records if r.get("country_stable") is not None]
    country_stable = sum(1 for r in country_stable_known if r.get("country_stable"))
    city_backtrack = sum(1 for r in records if r.get("city_backtrack_conflicts"))
    street_backtrack = sum(1 for r in records if r.get("street_backtrack_conflicts"))
    has_soft_conflict_fields = any(
        "city_soft_conflicts" in r or "street_soft_conflicts" in r
        for r in records
    )
    city_soft_conflict = sum(1 for r in records if r.get("city_soft_conflicts"))
    street_soft_conflict = sum(1 for r in records if r.get("street_soft_conflicts"))
    descent_blocked = [r for r in records if r.get("country_descent_blocked_reason")]
    descent_block_reasons = Counter(
        r.get("country_descent_blocked_reason") for r in descent_blocked
    )

    pred_continent_counts = Counter()
    pred_continent_correct = defaultdict(int)
    gt_continent_counts = Counter()
    gt_continent_correct = defaultdict(int)
    confusion = defaultdict(Counter)
    north_america_false_positives = []

    for record in records:
        pred_cont = continent_of(canonicalize_country(record.get("pred_country") or "") or "")
        if pred_cont:
            pred_continent_counts[pred_cont] += 1
            if float(record.get("dist_km", float("inf"))) <= 2500:
                pred_continent_correct[pred_cont] += 1

        gt_cont = _record_gt_continent(record)
        if gt_cont:
            gt_continent_counts[gt_cont] += 1
            if float(record.get("dist_km", float("inf"))) <= 2500:
                gt_continent_correct[gt_cont] += 1

        if pred_cont and gt_cont:
            confusion[pred_cont][gt_cont] += 1
            if pred_cont == "North America" and gt_cont != "North America":
                north_america_false_positives.append(record)

    na_fp_by_gt = Counter(
        _record_gt_continent(r) or "Unknown" for r in north_america_false_positives
    )
    na_fp_by_country = Counter(
        canonicalize_country(r.get("pred_country") or "") or r.get("pred_country") or "Unknown"
        for r in north_america_false_positives
    )
    na_fp_examples = sorted(
        north_america_false_positives,
        key=lambda r: float(r.get("dist_km", 0.0)),
        reverse=True,
    )[:10]

    return {
        "total": total,
        "accuracy": {
            str(thr): round(100.0 * correct[thr] / total, 2) if total else 0.0
            for thr in EVAL_THRESHOLDS
        },
        "unknown_country_rate": round(100.0 * unknown / total, 2) if total else 0.0,
        "country_top_mass": {
            "mean": round(mean(masses), 4) if masses else None,
            "median": round(median(masses), 4) if masses else None,
        },
        "geocode_source": dict(source_counts),
        "country_consistency": dict(consistency_counts),
        "country_child_conflict_rate": round(100.0 * len(conflicts) / total, 2) if total else 0.0,
        "country_replaced_rate": round(100.0 * country_replaced / total, 2) if total else 0.0,
        "country_web_enhanced_rate": round(100.0 * country_web_enhanced / total, 2) if total else 0.0,
        "country_visual_delta": {
            "mean": round(mean(visual_deltas), 4) if visual_deltas else None,
            "median": round(median(visual_deltas), 4) if visual_deltas else None,
        },
        "country_web_delta": {
            "mean": round(mean(web_deltas), 4) if web_deltas else None,
            "median": round(median(web_deltas), 4) if web_deltas else None,
        },
        "country_stable_rate": (
            round(100.0 * country_stable / len(country_stable_known), 2)
            if country_stable_known else None
        ),
        "backtrack_conflict_rate": {
            "city": round(100.0 * city_backtrack / total, 2) if total else 0.0,
            "street": round(100.0 * street_backtrack / total, 2) if total else 0.0,
        },
        "soft_conflict_rate": ({
            "city": round(100.0 * city_soft_conflict / total, 2) if total else 0.0,
            "street": round(100.0 * street_soft_conflict / total, 2) if total else 0.0,
        } if has_soft_conflict_fields else None),
        "country_descent_blocked_rate": round(100.0 * len(descent_blocked) / total, 2) if total else 0.0,
        "country_descent_blocked_reasons": dict(descent_block_reasons),
        "diagnostic_buckets": {
            "country_top_mass": _bucket_accuracy(records, _mass_bucket),
            "country_stable": _bucket_accuracy(records, lambda r: r.get("country_stable")),
            "geocode_source": _bucket_accuracy(records, lambda r: r.get("geocode_source") or "missing"),
            "country_web_enhanced": _bucket_accuracy(records, lambda r: bool(r.get("country_web_enhanced"))),
            "country_replaced": _bucket_accuracy(records, lambda r: bool(r.get("country_replaced"))),
            "country_descent_blocked_reason": _bucket_accuracy(
                records,
                lambda r: r.get("country_descent_blocked_reason") or "not_blocked",
            ),
        },
        "predicted_continent_breakdown": {
            cont: {
                "n": n,
                "continent_acc": round(100.0 * pred_continent_correct[cont] / n, 2),
            }
            for cont, n in sorted(pred_continent_counts.items())
        },
        "gt_continent_breakdown": {
            cont: {
                "n": n,
                "continent_acc": round(100.0 * gt_continent_correct[cont] / n, 2),
            }
            for cont, n in sorted(gt_continent_counts.items())
        },
        "predicted_vs_gt_continent": {
            pred: dict(sorted(gt_counts.items()))
            for pred, gt_counts in sorted(confusion.items())
        },
        "north_america_false_positives": {
            "n": len(north_america_false_positives),
            "by_gt_continent": dict(sorted(na_fp_by_gt.items())),
            "by_pred_country": dict(na_fp_by_country.most_common(10)),
            "examples": [
                {
                    "photo_id": r.get("photo_id"),
                    "gt_lat": r.get("gt_lat"),
                    "gt_lon": r.get("gt_lon"),
                    "gt_continent": _record_gt_continent(r),
                    "pred_country": r.get("pred_country"),
                    "pred_city": r.get("pred_city"),
                    "pred_street": r.get("pred_street"),
                    "dist_km": round(float(r.get("dist_km", 0.0)), 1),
                }
                for r in na_fp_examples
            ],
        },
    }


def _print_report(report: dict) -> None:
    print(f"Total records: {report['total']}")
    print("\nAccuracy")
    for thr in EVAL_THRESHOLDS:
        print(f"  {_THRESHOLD_LABELS[thr]:>18}: {report['accuracy'][str(thr)]:6.2f}%")

    print(f"\nUnknown country rate: {report['unknown_country_rate']:.2f}%")
    mass = report["country_top_mass"]
    print(f"Country posterior top mass: mean={mass['mean']} median={mass['median']}")
    print(f"Country-child conflict rate: {report['country_child_conflict_rate']:.2f}%")
    print(f"Country replace rate: {report['country_replaced_rate']:.2f}%")
    print(f"Country web enhance rate: {report['country_web_enhanced_rate']:.2f}%")
    visual_delta = report.get("country_visual_delta", {})
    if visual_delta.get("mean") is not None:
        print(
            "Country visual delta: "
            f"mean={visual_delta['mean']} median={visual_delta['median']}"
        )
    web_delta = report.get("country_web_delta", {})
    if web_delta.get("mean") is not None:
        print(
            "Country web delta: "
            f"mean={web_delta['mean']} median={web_delta['median']}"
        )
    if report["country_stable_rate"] is not None:
        print(f"Country stable rate: {report['country_stable_rate']:.2f}%")
    backtrack = report["backtrack_conflict_rate"]
    print(
        "Backtrack conflict rate: "
        f"city={backtrack['city']:.2f}% street={backtrack['street']:.2f}%"
    )
    soft_conflict = report.get("soft_conflict_rate", {})
    if soft_conflict:
        print(
            "Soft conflict rate: "
            f"city={soft_conflict['city']:.2f}% street={soft_conflict['street']:.2f}%"
        )
    print(f"Country descent blocked rate: {report['country_descent_blocked_rate']:.2f}%")
    if report["country_descent_blocked_reasons"]:
        print("Country descent blocked reasons")
        for key, value in sorted(report["country_descent_blocked_reasons"].items()):
            print(f"  {key}: {value}")

    print("\nGeocode source")
    for key, value in sorted(report["geocode_source"].items()):
        print(f"  {key}: {value}")

    print("\nCountry consistency")
    for key, value in sorted(report["country_consistency"].items()):
        print(f"  {key}: {value}")

    print("\nPredicted continent breakdown")
    for cont, stats in report["predicted_continent_breakdown"].items():
        print(f"  {cont}: n={stats['n']} acc@2500={stats['continent_acc']:.2f}%")

    if report["gt_continent_breakdown"]:
        print("\nGT continent breakdown")
        for cont, stats in report["gt_continent_breakdown"].items():
            print(f"  {cont}: n={stats['n']} acc@2500={stats['continent_acc']:.2f}%")

    if report["predicted_vs_gt_continent"]:
        print("\nPredicted vs GT continent")
        for pred, gt_counts in report["predicted_vs_gt_continent"].items():
            details = ", ".join(f"{gt}:{n}" for gt, n in gt_counts.items())
            print(f"  pred={pred}: {details}")

    na_fp = report["north_america_false_positives"]
    print(f"\nNorth America false positives: n={na_fp['n']}")
    if na_fp["by_gt_continent"]:
        print("  by GT continent: " + ", ".join(
            f"{k}:{v}" for k, v in na_fp["by_gt_continent"].items()
        ))
    if na_fp["by_pred_country"]:
        print("  by predicted country: " + ", ".join(
            f"{k}:{v}" for k, v in na_fp["by_pred_country"].items()
        ))

    buckets = report.get("diagnostic_buckets", {})
    if buckets:
        print("\nDiagnostic buckets (Country <750km / Continent <2500km)")
        for bucket_name in ("country_top_mass", "geocode_source", "country_descent_blocked_reason"):
            bucket = buckets.get(bucket_name, {})
            if not bucket:
                continue
            print(f"  {bucket_name}")
            for key, stats in bucket.items():
                acc = stats["accuracy"]
                print(
                    f"    {key}: n={stats['n']} "
                    f"country={acc.get('750', 0.0):.2f}% continent={acc.get('2500', 0.0):.2f}%"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to evaluate.py result JSON")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    with open(args.pred) as f:
        data = json.load(f)
    report = analyze(data.get("records", []))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
