"""
Collects Warsaw apartment listings from otodom.pl and enriches them with
POI distances from OpenStreetMap (Overpass API) to match the model's feature set.

Usage:
    python -m src.data_collection.collect_data
    python -m src.data_collection.collect_data --pages 20 --no-enrich
    python -m src.data_collection.collect_data --output data/raw/fresh.parquet
"""

import argparse
import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

WARSAW_CENTER = (52.2297, 21.0122)
WARSAW_BBOX = (52.09, 20.85, 52.37, 21.27)  # south, west, north, east

OTODOM_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.otodom.pl/",
}

# Overpass filter → feature name
POI_FILTERS = {
    "schoolDistance":       '["amenity"="school"]',
    "clinicDistance":       '["amenity"~"clinic|hospital|doctors"]',
    "postOfficeDistance":   '["amenity"="post_office"]',
    "kindergartenDistance": '["amenity"="kindergarten"]',
    "restaurantDistance":   '["amenity"~"restaurant|cafe|fast_food"]',
    "collegeDistance":      '["amenity"~"college|university"]',
    "pharmacyDistance":     '["amenity"="pharmacy"]',
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_km(lat: float, lon: float, coords: list[tuple[float, float]]) -> float:
    if not coords:
        return np.nan
    return min(haversine(lat, lon, plat, plon) for plat, plon in coords)


# ---------------------------------------------------------------------------
# Otodom scraping
# ---------------------------------------------------------------------------

def fetch_page(page: int, session: requests.Session) -> dict | None:
    params = {"page": page} if page > 1 else {}
    try:
        resp = session.get(OTODOM_URL, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Page %d fetch failed: %s", page, e)
        return None

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        resp.text,
        re.DOTALL,
    )
    if not match:
        log.warning("__NEXT_DATA__ not found on page %d — site structure may have changed", page)
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        log.warning("JSON parse error on page %d: %s", page, e)
        return None


def _find_ads(page_props: dict) -> dict | None:
    """Try known key paths across different Otodom site versions."""
    candidates = [
        page_props.get("data", {}).get("searchAds"),
        page_props.get("data", {}).get("listing"),
        page_props.get("listing", {}).get("listing"),
        page_props.get("searchAds"),
    ]
    return next((c for c in candidates if c is not None), None)


def extract_listings(next_data: dict) -> tuple[list[dict], int]:
    """Returns (raw listing dicts, total_pages)."""
    try:
        page_props = next_data["props"]["pageProps"]
    except KeyError:
        log.warning("Unexpected __NEXT_DATA__ root structure")
        return [], 0

    ads = _find_ads(page_props)
    if ads is None:
        log.warning("Could not locate ads block — available keys: %s", list(page_props.keys()))
        return [], 0

    items = ads.get("items") or ads.get("ads") or []
    pagination = ads.get("pagination") or {}
    total_pages = int(pagination.get("totalPages") or pagination.get("pageCount") or 1)

    listings = []
    for item in items:
        loc = item.get("location") or {}
        coords = loc.get("coordinates") or {}
        lat = coords.get("latitude")
        lon = coords.get("longitude")

        price_obj = item.get("totalPrice") or item.get("price") or {}
        price = price_obj.get("value") if isinstance(price_obj, dict) else price_obj

        area = item.get("areaInSquareMeters") or item.get("area")

        floors_obj = item.get("floors") or {}
        floor = item.get("floorNumber") or (
            floors_obj.get("floor") if isinstance(floors_obj, dict) else None
        )
        floor_count = item.get("totalFloors") or (
            floors_obj.get("total") if isinstance(floors_obj, dict) else None
        )

        build_year = item.get("buildYear") or (
            (item.get("construction") or {}).get("yearBuilt")
        )

        if None in (lat, lon, price, area):
            continue

        listings.append({
            "price": float(price),
            "squareMeters": float(area),
            "floor": float(floor) if floor is not None else np.nan,
            "floorCount": float(floor_count) if floor_count is not None else np.nan,
            "buildYear": float(build_year) if build_year is not None else np.nan,
            "latitude": float(lat),
            "longitude": float(lon),
        })

    return listings, total_pages


# ---------------------------------------------------------------------------
# POI enrichment via Overpass API
# ---------------------------------------------------------------------------

def _overpass_query(amenity_filter: str) -> list[tuple[float, float]]:
    south, west, north, east = WARSAW_BBOX
    query = (
        f"[out:json][timeout:60];\n"
        f"(\n"
        f"  node{amenity_filter}({south},{west},{north},{east});\n"
        f"  way{amenity_filter}({south},{west},{north},{east});\n"
        f");\n"
        f"out center;\n"
    )
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=90)
        resp.raise_for_status()
        coords = []
        for el in resp.json().get("elements", []):
            if "lat" in el:
                coords.append((el["lat"], el["lon"]))
            elif "center" in el:
                coords.append((el["center"]["lat"], el["center"]["lon"]))
        return coords
    except Exception as e:
        log.warning("Overpass query failed (%s): %s", amenity_filter, e)
        return []


def enrich_pois(listings: list[dict]) -> list[dict]:
    log.info("Fetching POI data from Overpass API (%d categories)...", len(POI_FILTERS))
    poi_data: dict[str, list[tuple[float, float]]] = {}

    for field, amenity_filter in POI_FILTERS.items():
        log.info("  %s...", field)
        poi_data[field] = _overpass_query(amenity_filter)
        log.info("    %d POIs", len(poi_data[field]))
        time.sleep(2)

    all_coords = [c for coords in poi_data.values() for c in coords]

    for listing in listings:
        lat, lon = listing["latitude"], listing["longitude"]
        for field, coords in poi_data.items():
            listing[field] = nearest_km(lat, lon, coords)
        listing["poiCount"] = sum(
            1 for plat, plon in all_coords if haversine(lat, lon, plat, plon) <= 1.0
        )

    return listings


# ---------------------------------------------------------------------------
# Date encoding (matches original dataset encoding)
# ---------------------------------------------------------------------------

_DATE_MAP = {
    (2024, 1): 0, (2024, 2): 1, (2024, 3): 2, (2024, 4): 3,
    (2024, 5): 4, (2024, 6): 5,
    (2023, 8): 6, (2023, 9): 7, (2023, 10): 8, (2023, 11): 9, (2023, 12): 10,
}


def encode_date(dt: datetime) -> int:
    return _DATE_MAP.get((dt.year, dt.month), 5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect(pages: int = 10, enrich: bool = True, output: Path = None) -> Path | None:
    if output is None:
        output = Path(__file__).parent.parent.parent / "data" / "raw" / "otodom_collected.parquet"

    session = requests.Session()
    all_listings: list[dict] = []
    total_pages: int | None = None

    for page in range(1, pages + 1):
        if total_pages is not None and page > total_pages:
            log.info("Reached last available page (%d)", total_pages)
            break

        log.info("Fetching page %d/%s...", page, total_pages or "?")
        next_data = fetch_page(page, session)
        if next_data is None:
            continue

        listings, tp = extract_listings(next_data)

        if total_pages is None and tp:
            total_pages = tp
            log.info("Total pages on otodom: %d", total_pages)

        log.info("  %d listings extracted", len(listings))
        all_listings.extend(listings)

        delay = 1.5 + (page % 3) * 0.5  # 1.5–2.5 s, varied to look less bot-like
        time.sleep(delay)

    log.info("Total collected: %d listings", len(all_listings))

    if not all_listings:
        log.error("No listings collected — otodom page structure may have changed")
        return None

    if enrich:
        all_listings = enrich_pois(all_listings)
    else:
        for listing in all_listings:
            for field in POI_FILTERS:
                listing.setdefault(field, np.nan)
            listing.setdefault("poiCount", np.nan)

    now = datetime.now()
    for listing in all_listings:
        listing["centreDistance"] = haversine(
            listing["latitude"], listing["longitude"], *WARSAW_CENTER
        )
        listing["date"] = encode_date(now)

    df = pd.DataFrame(all_listings)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    log.info("Saved to %s", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Warsaw apartment listings from otodom.pl")
    parser.add_argument("--pages", type=int, default=10, help="Pages to scrape (default: 10, ~360 listings)")
    parser.add_argument("--no-enrich", action="store_true", help="Skip POI enrichment (faster, less accurate)")
    parser.add_argument("--output", type=Path, default=None, help="Output .parquet path")
    args = parser.parse_args()

    collect(pages=args.pages, enrich=not args.no_enrich, output=args.output)
