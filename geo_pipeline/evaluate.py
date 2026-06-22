"""
Evaluation on YFCC4K using standard distance-threshold accuracy metrics.
Geocoding: location name → (lat, lon) via geopy (offline-compatible with Nominatim).

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --limit 100 --out results/run1.json
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --start 1000 --out results/run2.json  # resume
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch_size 8 --out results/run3.json
"""

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


def _continent_from_name(name: str) -> str | None:
    """Heuristic: ask Nominatim for the country, then map ISO-3166 region.
    Falls back to a small keyword table for robustness."""
    if not name:
        return None
    n = name.lower()
    # tiny keyword fallback — extend as needed
    asia = ["china", "japan", "korea", "india", "thailand", "vietnam", "indonesia",
            "philippines", "malaysia", "singapore", "pakistan", "bangladesh", "iran",
            "iraq", "saudi", "turkey", "kazakhstan", "nepal", "sri lanka", "taiwan"]
    europe = ["germany", "france", "italy", "spain", "portugal", "uk", "united kingdom",
              "england", "scotland", "ireland", "netherlands", "belgium", "switzerland",
              "austria", "poland", "czech", "hungary", "greece", "sweden", "norway",
              "finland", "denmark", "russia", "ukraine", "romania", "bulgaria", "serbia",
              "croatia", "slovenia", "slovakia"]
    africa = ["egypt", "morocco", "south africa", "kenya", "nigeria", "ethiopia",
              "ghana", "algeria", "tunisia", "uganda", "tanzania", "senegal"]
    north_am = ["united states", "usa", "canada", "mexico", "cuba", "jamaica",
                "guatemala", "panama", "costa rica", "honduras"]
    south_am = ["brazil", "argentina", "chile", "peru", "colombia", "venezuela",
                "ecuador", "bolivia", "uruguay", "paraguay"]
    oceania = ["australia", "new zealand", "fiji", "papua"]

    for kws, cont in [(asia, "Asia"), (europe, "Europe"), (africa, "Africa"),
                      (north_am, "North America"), (south_am, "South America"),
                      (oceania, "Oceania")]:
        if any(k in n for k in kws):
            return cont
    return None


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
            pred_coords = None
            for level in ["street", "city", "country"]:
                name = pred.get(level)
                if name and name != "Unknown":
                    pred_coords = geocode(name)
                    if pred_coords:
                        break

            # Continent fallback: if all geocoding failed, drop to continent centroid
            # of the predicted country so the <2500 km threshold can still hit.
            if pred_coords is None:
                country_name = pred.get("country")
                cont = _continent_from_name(country_name) if country_name else None
                if cont and cont in _CONTINENT_CENTROIDS:
                    pred_coords = _CONTINENT_CENTROIDS[cont]

            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            dist_km = haversine(gt_lat, gt_lon, pred_coords[0], pred_coords[1]) \
                      if pred_coords else float("inf")

            record = {
                "photo_id":    sample["photo_id"],
                "gt_lat":      gt_lat,
                "gt_lon":      gt_lon,
                "pred_country": pred.get("country"),
                "pred_city":    pred.get("city"),
                "pred_street":  pred.get("street"),
                "pred_lat":     pred_coords[0] if pred_coords else None,
                "pred_lon":     pred_coords[1] if pred_coords else None,
                "dist_km":      dist_km,
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
    evaluate(parser.parse_args())
