"""
Evaluation on YFCC4K using standard distance-threshold accuracy metrics.
Geocoding: location name → (lat, lon) via geopy (offline-compatible with Nominatim).

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --limit 100 --out results/run1.json
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --start 1000 --out results/run2.json  # resume
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch_size 8 --out results/run3.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderRateLimited

from models.mllm_client import MLLMClient
from pipeline import GeoPipeline
from data.yfcc4k_loader import YFCC4KDataset
from country_aliases import COUNTRY_TO_CONTINENT, continent_of, canonicalize_country
from config import EVAL_THRESHOLDS, YFCC4K_IMG_DIR, YFCC4K_GPS_CSV


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


_geocoder = Nominatim(user_agent="geo_pipeline_eval", timeout=10)


# Continent centroids — used as a last-resort fallback so the continent
# threshold (<2500 km) can still hit even when country/city/street geocoding
# all fail. Coordinates are rough geographic centers.
_CONTINENT_CENTROIDS = {
    "Africa":        (1.65,   17.83),
    "Asia":          (34.05, 100.62),
    "Europe":        (54.53,  15.26),
    "North America": (54.53, -105.26),
    "South America": (-8.78,  -55.49),
    "Oceania":       (-22.74, 140.02),
    "Antarctica":    (-82.86,  21.00),
}


# Country name → continent map + canonicalizer live in country_aliases.py
# (shared with pipeline.py). Import re-exports above.


def geocode(location_name: str):
    """Name → (lat, lon). Returns None if lookup fails."""
    try:
        loc = _geocoder.geocode(location_name)
        time.sleep(1.1)  # Nominatim enforces 1 req/sec
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        pass
    except GeocoderRateLimited:
        time.sleep(5)
    return None


def _geocode_level(
    name: str,
    level: str,
    country: str | None,
    strict_child_geocode: bool = False,
    allow_bare_city_geocode: bool = True,
):
    """Geocode one prediction level and return coords plus diagnostic source.

    Nominatim's gazetteer is ambiguous for many city/street names (a dozen
    "Springfield"s, two "Naples", etc.). When the predicted country is
    available, qualifying street/city queries first shrinks the search space.
    Bare street/city fallback remains enabled by default for comparability with
    v5/v6; strict mode disables all unqualified child fallback.
    """
    if level in ("street", "city"):
        country_ok = bool(country and country.lower() not in ("unknown", ""))
        name_has_country = country_ok and country.lower() in name.lower()
        embedded_country = canonicalize_country(name)
        if country_ok and embedded_country and embedded_country != country.lower():
            return None, None, "child_country_conflict"

        if country_ok and not name_has_country:
            coords = geocode(f"{name}, {country}")
            if coords is not None:
                return coords, f"{level}_country_qualified", "country_qualified"

        if strict_child_geocode and not name_has_country:
            return None, None, "failed"
        if level == "city" and not name_has_country and not allow_bare_city_geocode:
            return None, None, "failed"

        coords = geocode(name)
        if coords is not None:
            consistency = "country_in_name" if name_has_country else "unchecked"
            return coords, f"{level}_bare", consistency
        return None, None, "failed"

    coords = geocode(name)
    if coords is not None:
        return coords, "country", "country_level"
    return None, None, "failed"


def _continent_fallback_coords(pred: dict) -> tuple | None:
    """Vote a continent from the top-3 country posterior and return its centroid.

    Used as a last resort when Nominatim returns None for every level. The
    posterior is informative even when the argmax country name doesn't geocode
    (e.g. argmax='Burma', no Nominatim hit, but Asia is still right). Returns
    None if the country posterior is empty or no candidate maps to a continent.
    """
    country_post = pred.get("country_posterior", {}) or {}
    if not country_post:
        # try the bare country name as a last resort
        cont = continent_of(pred.get("country") or "")
        return _CONTINENT_CENTROIDS.get(cont) if cont else None

    sorted_countries = sorted(country_post.items(), key=lambda kv: -kv[1])[:3]
    votes: dict[str, float] = {}
    for cname, p in sorted_countries:
        cont = continent_of(cname)
        if cont:
            votes[cont] = votes.get(cont, 0.0) + p
    if not votes:
        return None
    best = max(votes, key=votes.get)
    return _CONTINENT_CENTROIDS.get(best)


def evaluate(args):
    mllm     = MLLMClient()
    pipeline = GeoPipeline(mllm)
    dataset  = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)

    start = args.start or 0
    end   = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))
    batch_size = args.batch_size

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    total   = 0

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Evaluating batches"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        try:
            preds = pipeline.predict_batch([s["image"] for s in samples])
        except Exception as e:
            print(f"[WARN] batch {batch_start} failed: {e}")
            continue

        for sample, pred in zip(samples, preds):
            pred_country = pred.get("country")
            # If the hypothesize step failed to name a real country (parser
            # returned "Unknown" or a non-country string like "Southeast Asia"),
            # try to salvage one from the city/street strings — they very often
            # look like "Toronto, Canada" or "Paris, France". This is EVALUATE-
            # ONLY: we never feed the fallback back into country_posterior, so
            # DST/POMDP behaviour is unchanged. Rescues ~73% of Unknown records
            # per full_v4 offline check.
            if not pred_country or canonicalize_country(pred_country) is None:
                for field in ("city", "street"):
                    salvaged = canonicalize_country(pred.get(field) or "")
                    if salvaged:
                        pred_country = salvaged
                        break

            pred_coords = None
            geocode_source = None
            country_consistency = None
            # Hierarchical geocode: street → city → country. For street/city
            # try country-qualified queries before bare-name fallback.
            for level in ["street", "city", "country"]:
                name = pred.get(level)
                if name and name != "Unknown":
                    qualifier = pred_country if level in ("street", "city") else None
                    pred_coords, geocode_source, country_consistency = _geocode_level(
                        name,
                        level,
                        qualifier,
                        args.strict_child_geocode,
                        args.allow_bare_city_geocode,
                    )
                    if pred_coords:
                        break
            # Last-resort continent centroid (saves the <=2500km threshold when
            # Nominatim returns None for every level, e.g. obsolete country names).
            if pred_coords is None:
                pred_coords = _continent_fallback_coords(pred)
                if pred_coords is not None:
                    geocode_source = "continent_fallback"
                    country_consistency = "fallback"

            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            dist_km = haversine(gt_lat, gt_lon, pred_coords[0], pred_coords[1]) \
                      if pred_coords else float("inf")

            record = {
                "photo_id":    sample["photo_id"],
                "gt_lat":      gt_lat,
                "gt_lon":      gt_lon,
                "pred_country": pred_country,
                "pred_city":    pred.get("city"),
                "pred_street":  pred.get("street"),
                "pred_lat":     pred_coords[0] if pred_coords else None,
                "pred_lon":     pred_coords[1] if pred_coords else None,
                "dist_km":      dist_km,
                "geocode_source": geocode_source,
                "country_consistency": country_consistency,
                "country_posterior": {
                    k: round(float(v), 4)
                    for k, v in (pred.get("country_posterior") or {}).items()
                },
                "country_stable": pred.get("country_stable"),
                "city_stable": pred.get("city_stable"),
                "street_stable": pred.get("street_stable"),
                "country_replaced": bool(pred.get("country_replaced")),
                "country_web_enhanced": bool(pred.get("country_web_enhanced")),
                "country_web_search_query": pred.get("country_web_search_query"),
                "country_visual_delta": pred.get("country_visual_delta"),
                "country_web_delta": pred.get("country_web_delta"),
                "country_descent_blocked_reason": pred.get("country_descent_blocked_reason"),
                "city_backtrack_conflicts": pred.get("city_backtrack_conflicts", []),
                "street_backtrack_conflicts": pred.get("street_backtrack_conflicts", []),
                "raw_country_response": pred.get("country_raw_response"),
                "raw_city_response":    pred.get("city_raw_response"),
                "raw_street_response":  pred.get("street_raw_response"),
            }
            records.append(record)
            total += 1
            for thr in EVAL_THRESHOLDS:
                if dist_km <= thr:
                    correct[thr] += 1

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nResults on YFCC4K ({total} images, indices {start}–{start+total-1})")
    print(f"{'Threshold':>12}  {'Accuracy':>10}")
    print("-" * 26)
    for thr in EVAL_THRESHOLDS:
        label = {1: "Street <1km", 25: "City <25km", 200: "Region <200km",
                 750: "Country <750km", 2500: "Continent <2500km"}[thr]
        acc = 100.0 * correct[thr] / total if total > 0 else 0.0
        print(f"{label:>14}  {acc:>9.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "summary": {str(k): round(100 * v / total, 2) if total else 0
                        for k, v in correct.items()},
            "total": total,
            "start": start,
            "records": records,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv",    default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit",      type=int, default=None, help="max images to evaluate")
    parser.add_argument("--start",      type=int, default=0,    help="start from this dataset index (for resuming)")
    parser.add_argument("--batch_size", type=int, default=20,   help="images per GPU batch")
    parser.add_argument("--out",        default="results/eval.json")
    parser.add_argument(
        "--strict_child_geocode",
        action="store_true",
        help="Disable unqualified street/city Nominatim matches for strict consistency ablations.",
    )
    parser.add_argument(
        "--allow_bare_city_geocode",
        action="store_true",
        default=True,
        help="Allow unqualified city Nominatim matches; enabled by default for v5 comparability.",
    )
    parser.add_argument(
        "--disable_bare_city_geocode",
        action="store_false",
        dest="allow_bare_city_geocode",
        help="Disable unqualified city Nominatim matches for ablations.",
    )
    evaluate(parser.parse_args())
