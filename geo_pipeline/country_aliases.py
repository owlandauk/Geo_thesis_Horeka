"""
Country name → continent map + canonicalization helpers.

Shared between pipeline.py (used to normalize raw MLLM country labels before
they enter _softmax_prior — earlier full-run diagnosis showed many records got
Unknown because the model emitted "USA"/"California, USA"/"Southeast Asia"
that no downstream step recognized) and evaluate.py (used for the
continent-centroid geocode fallback and, as of full_v5, for extracting
country names out of pred_city / pred_street strings).
"""
from __future__ import annotations

# Country name → continent. Covers the high-prevalence YFCC4K countries plus
# common alternate names (Burma/Myanmar, Holland/Netherlands, etc.) that
# Nominatim sometimes mishandles.
COUNTRY_TO_CONTINENT = {
    # Asia
    "china": "Asia", "japan": "Asia", "south korea": "Asia", "korea": "Asia",
    "north korea": "Asia", "india": "Asia", "pakistan": "Asia", "bangladesh": "Asia",
    "thailand": "Asia", "vietnam": "Asia", "indonesia": "Asia", "malaysia": "Asia",
    "singapore": "Asia", "philippines": "Asia", "taiwan": "Asia", "myanmar": "Asia",
    "burma": "Asia", "cambodia": "Asia", "laos": "Asia", "nepal": "Asia",
    "sri lanka": "Asia", "mongolia": "Asia", "kazakhstan": "Asia",
    "uzbekistan": "Asia", "iran": "Asia", "iraq": "Asia", "saudi arabia": "Asia",
    "uae": "Asia", "israel": "Asia", "turkey": "Asia", "jordan": "Asia",
    "lebanon": "Asia", "syria": "Asia", "qatar": "Asia", "kuwait": "Asia",
    "oman": "Asia", "afghanistan": "Asia",
    # Europe
    "germany": "Europe", "france": "Europe", "italy": "Europe", "spain": "Europe",
    "portugal": "Europe", "united kingdom": "Europe", "uk": "Europe",
    "great britain": "Europe", "england": "Europe", "scotland": "Europe",
    "wales": "Europe", "ireland": "Europe", "netherlands": "Europe",
    "holland": "Europe", "belgium": "Europe", "switzerland": "Europe",
    "austria": "Europe", "poland": "Europe", "czech republic": "Europe",
    "czechia": "Europe", "hungary": "Europe", "greece": "Europe",
    "sweden": "Europe", "norway": "Europe", "finland": "Europe", "denmark": "Europe",
    "iceland": "Europe", "russia": "Europe", "ukraine": "Europe",
    "romania": "Europe", "bulgaria": "Europe", "serbia": "Europe",
    "croatia": "Europe", "slovenia": "Europe", "slovakia": "Europe",
    "estonia": "Europe", "latvia": "Europe", "lithuania": "Europe",
    "luxembourg": "Europe", "malta": "Europe", "cyprus": "Europe",
    "albania": "Europe", "bosnia": "Europe", "macedonia": "Europe",
    # Africa
    "egypt": "Africa", "morocco": "Africa", "south africa": "Africa",
    "kenya": "Africa", "nigeria": "Africa", "ethiopia": "Africa",
    "ghana": "Africa", "algeria": "Africa", "tunisia": "Africa",
    "uganda": "Africa", "tanzania": "Africa", "senegal": "Africa",
    "zimbabwe": "Africa", "namibia": "Africa", "botswana": "Africa",
    "madagascar": "Africa", "libya": "Africa", "sudan": "Africa",
    "ivory coast": "Africa", "cameroon": "Africa", "angola": "Africa",
    "mozambique": "Africa", "zambia": "Africa", "rwanda": "Africa",
    # North America
    "united states": "North America", "usa": "North America",
    "us": "North America", "america": "North America",
    "canada": "North America", "mexico": "North America", "cuba": "North America",
    "jamaica": "North America", "guatemala": "North America",
    "panama": "North America", "costa rica": "North America",
    "honduras": "North America", "nicaragua": "North America",
    "el salvador": "North America", "dominican republic": "North America",
    "haiti": "North America", "puerto rico": "North America",
    # South America
    "brazil": "South America", "argentina": "South America",
    "chile": "South America", "peru": "South America",
    "colombia": "South America", "venezuela": "South America",
    "ecuador": "South America", "bolivia": "South America",
    "uruguay": "South America", "paraguay": "South America",
    "guyana": "South America", "suriname": "South America",
    # Oceania
    "australia": "Oceania", "new zealand": "Oceania", "fiji": "Oceania",
    "papua new guinea": "Oceania", "samoa": "Oceania", "tonga": "Oceania",
}


# Sorted longest-first so canonicalize_country() picks "united kingdom" over
# "united" or "kingdom" when a raw label contains both.
_ALIASES_BY_LENGTH = sorted(COUNTRY_TO_CONTINENT.keys(), key=len, reverse=True)


# Multiple aliases in COUNTRY_TO_CONTINENT refer to the same country
# ("usa" / "us" / "america" / "united states"). Without this second map,
# canonicalize_country("USA") returns "usa" and canonicalize_country
# ("United States") returns "united states" — _collect_scores then treats
# them as two candidates and splits the softmax mass across duplicates.
# Map every alias to its canonical (Nominatim-friendly) country name here.
_ALIAS_TO_CANONICAL = {
    # North America
    "usa": "united states", "us": "united states", "america": "united states",
    # Europe
    "uk": "united kingdom", "great britain": "united kingdom",
    "england": "united kingdom", "scotland": "united kingdom",
    "wales": "united kingdom",
    "holland": "netherlands",
    "czechia": "czech republic",
    # Asia
    "burma": "myanmar",
    "korea": "south korea",
}


def continent_of(country: str) -> str | None:
    if not country:
        return None
    return COUNTRY_TO_CONTINENT.get(country.strip().lower())


def canonicalize_country(raw: str) -> str | None:
    """Return the canonical country name if any alias appears in `raw`, else None.

    Handles: "USA" → "united states", "California, USA" → "united states",
    "Southeast Asia" → None, "UK" → "united kingdom".
    Prefers the last comma-separated token first ("Toronto, Canada" pattern),
    then falls back to a longest-match substring scan. Aliases collapse to a
    single canonical form so _softmax_prior sees one entry per country.
    """
    if not raw:
        return None
    low = raw.strip().lower()

    def _canon(name: str) -> str:
        return _ALIAS_TO_CANONICAL.get(name, name)

    # exact hit — fast path
    if low in COUNTRY_TO_CONTINENT:
        return _canon(low)

    # last comma-separated tail: "City, Country" → "country"
    if "," in low:
        tail = low.rsplit(",", 1)[1].strip()
        if tail in COUNTRY_TO_CONTINENT:
            return _canon(tail)

    # longest-match substring scan
    for alias in _ALIASES_BY_LENGTH:
        if alias in low:
            return _canon(alias)
    return None
