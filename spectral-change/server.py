"""
Spectral Change — MCP Server
============================
Spectral indices from the Awesome Spectral Indices catalogue (spyndex), computed
on Sentinel-2 L2A from the Copernicus Data Space Ecosystem (CDSE), to answer
"how much changed here?" and "when did it change?".

Tools:
    search_spectral_indices  — find an index for a topic ("built-up", "flooding"). No network.
    get_spectral_index       — formula, bands, constants, reference of one index. No network.
    compute_index_change     — per-pixel median before vs after, delta, hectares of
                               increase / decrease, extent of the changed area, 4-panel map.
    compute_index_timeseries — cloud-masked area statistics for every Sentinel-2 scene,
                               monthly / yearly medians, trend, change point, plot + CSV.

Where things run:
    The index itself is computed by CDSE (Sentinel Hub) from an evalscript that this
    server generates from the spyndex formula. This server receives small rasters
    (change) or per-scene statistics (time series), computes hectares and summaries
    with numpy, and writes PNG / CSV / GeoTIFF files to CHANGE_MAPS_DIR
    (default: <tmp>/spectral_change).

Usage:
    python server.py --transport stdio
    python server.py --transport http        # http://localhost:8000/mcp

Credentials (CDSE OAuth client, https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings):
    headers X-CDSE-Client-Id / X-CDSE-Client-Secret, or env CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
    (.env in this folder or up to three parent folders). Only the two compute tools need them.
"""

from __future__ import annotations

import asyncio
import csv
import difflib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import spyndex
import tifffile
from asi_formula import (
    S2_L2A_BANDS,
    ConversionError,
    bbox_area_ha,
    build_median_evalscript,
    build_process_request,
    build_stats_evalscript,
    build_stats_request,
    change_extent,
    formula_to_js,
    grid_for_bbox,
    parse_bbox,
    parse_date,
    parse_stats_response,
    summarize_change,
    summarize_timeseries,
)
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from plots import change_figure, timeseries_figure

_SERVER_DIR = Path(__file__).resolve().parent
# This folder first, then parents (the hackathon repo keeps .env at its root).
# Never override variables already set in the process.
for _dir in [_SERVER_DIR, *list(_SERVER_DIR.parents)[:3]]:
    if (_dir / ".env").is_file():
        load_dotenv(_dir / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,  # stdio transport: never write logs to stdout
)
logger = logging.getLogger("spectral-change-mcp")

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
MAX_SERIES_DAYS = 3660  # ten years
TIMEOUT = 180.0
MAX_AREA_HA = 250_000.0  # 2,500 km²: keeps requests fast and inside the free CDSE quota
DEFAULT_SAVE_DIR = Path(os.getenv("CHANGE_MAPS_DIR", Path(tempfile.gettempdir()) / "spectral_change"))

METHODS = {
    "delta": "Per-pixel median index after minus median index before; threshold on the difference.",
}

# What a higher index value means, by ASI application domain.
DOMAIN_HINT = {
    "vegetation": "Higher = more or greener vegetation. A decrease usually means vegetation loss.",
    "water": "Higher = more water or moisture. An increase can mean flooding or wetter surfaces.",
    "urban": "Higher = more built-up or impervious surface (bare soil also scores high).",
    "burn": "Burn indices differ in sign: for NBR-type indices a decrease means burning; "
    "check the formula of this index.",
    "snow": "Higher = more snow or ice cover.",
    "soil": "Higher = more bare soil.",
    "clouds": "Cloud indices are not meant for change detection.",
}

mcp = FastMCP("Spectral Change", host="0.0.0.0", port=8000, stateless_http=True)

_CONSTANTS = {name: c.default for name, c in spyndex.constants.items()}


# ---------------------------------------------------------------------------
# Catalogue (no network)
# ---------------------------------------------------------------------------

_INDICES = spyndex.indices
_LOOKUP = {name.lower(): name for name in _INDICES}
DOMAINS = sorted({idx.application_domain for idx in _INDICES.values()})

# Words people use -> ASI application domain.
SYNONYMS = {
    "urban": "urban", "built": "urban", "builtup": "urban", "built-up": "urban", "building": "urban",
    "buildings": "urban", "city": "urban", "settlement": "urban", "impervious": "urban",
    "construction": "urban", "infrastructure": "urban", "sealed": "urban",
    "vegetation": "vegetation", "green": "vegetation", "greenness": "vegetation", "plant": "vegetation",
    "plants": "vegetation", "crop": "vegetation", "crops": "vegetation", "forest": "vegetation",
    "deforestation": "vegetation", "chlorophyll": "vegetation", "leaf": "vegetation", "biomass": "vegetation",
    "agriculture": "vegetation", "drought": "vegetation",
    "water": "water", "flood": "water", "flooding": "water", "lake": "water", "river": "water",
    "wetland": "water", "moisture": "water", "shoreline": "water", "coast": "water",
    "burn": "burn", "burned": "burn", "burnt": "burn", "fire": "burn", "wildfire": "burn", "scar": "burn",
    "snow": "snow", "ice": "snow", "glacier": "snow",
    "soil": "soil", "bare": "soil", "erosion": "soil", "desert": "soil",
    "cloud": "clouds", "clouds": "clouds", "haze": "clouds",
    "radar": "radar", "sar": "radar",
}

# Well-established indices are listed first within a domain. The canonical
# index of each domain gets the biggest boost, so "built-up" -> NDBI first.
CANONICAL = {"NDVI", "NDBI", "MNDWI", "NBR", "NDSI", "BI"}
POPULAR = {
    "EVI", "SAVI", "NDREI", "GNDVI", "MSAVI", "NDMI", "NDWI", "AWEInsh", "NBR2",
    "BAIS2", "UI", "IBI", "NBAI", "BLFEI", "NDII", "DBSI",
}


def _s2_status(name: str) -> tuple[bool, str | None]:
    idx = _INDICES[name]
    if idx.application_domain == "kernel":
        return False, "kernel index: needs a kernel function and parameters"
    try:
        formula_to_js(idx.formula, "sentinel-2-l2a", _CONSTANTS)
        return True, None
    except ConversionError as exc:
        return False, str(exc)


_S2_OK = {name: _s2_status(name) for name in _INDICES}


def _summary(name: str) -> dict:
    idx = _INDICES[name]
    return {
        "short_name": idx.short_name,
        "long_name": idx.long_name,
        "domain": idx.application_domain,
        "formula": idx.formula,
        "sentinel2": _S2_OK[name][0],
    }


@mcp.tool()
async def search_spectral_indices(
    query: str = "",
    application_domain: str | None = None,
    sentinel2_only: bool = True,
    limit: int = 10,
) -> str:
    """Find spectral indices in the Awesome Spectral Indices catalogue.

    Use this first when a question names a surface or process ("built-up area",
    "forest loss", "flooding", "burn scar", "snow cover") and you need to pick
    an index to compute. Results are ranked: exact name matches, then
    well-established indices (NDVI, NDBI, MNDWI, NBR...), then the rest.

    Args:
        query: Free text, e.g. "built-up", "water", "chlorophyll", "NDVI".
            Leave empty to list a domain.
        application_domain: Optional filter. One of: vegetation, water, urban,
            burn, snow, soil, clouds, radar, kernel.
        sentinel2_only: Only return indices that can be computed from
            Sentinel-2 bands (default true).
        limit: Maximum number of results (default 10, max 50).

    Returns:
        JSON with the matching indices (short_name, long_name, domain, formula,
        sentinel2), the total number of matches, and the domains searched.
        Pass a short_name to get_spectral_index for details, or to
        compute_index_change or compute_index_timeseries.
    """
    limit = max(1, min(int(limit), 50))
    if application_domain and application_domain not in DOMAINS:
        return json.dumps({
            "error": f"Unknown application_domain '{application_domain}'.",
            "domains": DOMAINS,
        })

    words = [w for w in re.split(r"[^a-z0-9\-]+", query.lower()) if w]
    domains = {SYNONYMS[w] for w in words if w in SYNONYMS}
    if application_domain:
        domains = {application_domain}

    scored = []
    for name, idx in _INDICES.items():
        if sentinel2_only and not _S2_OK[name][0]:
            continue
        if application_domain and idx.application_domain != application_domain:
            continue
        text = f"{idx.short_name} {idx.long_name}".lower()
        score = 0.0
        if query.strip().lower() == name.lower():
            score += 100
        if idx.application_domain in domains:
            score += 10
        score += sum(4 for w in words if len(w) > 2 and w in text)
        if not words and application_domain:
            score += 1
        if score <= 0:
            continue
        if name in CANONICAL:
            score += 9
        elif name in POPULAR:
            score += 3
        scored.append((score, name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    result = {
        "query": query,
        "domains_matched": sorted(domains),
        "total_matches": len(scored),
        "results": [_summary(name) for _, name in scored[:limit]],
    }
    if not scored:
        result["hint"] = (
            f"No match. Try a domain ({', '.join(DOMAINS)}) or words like "
            "'built-up', 'water', 'vegetation', 'burn', 'snow'."
        )
    return json.dumps(result)


@mcp.tool()
async def get_spectral_index(short_name: str) -> str:
    """Details of one spectral index: formula, bands, constants, reference.

    Args:
        short_name: Index name from the catalogue, e.g. "NDVI", "NDBI",
            "MNDWI", "EVI" (case-insensitive).

    Returns:
        JSON with short_name, long_name, application_domain, formula, bands
        (each ASI band code with its description and Sentinel-2 band name),
        constants with default values, reference (DOI or URL), platforms,
        sentinel2_computable and, if false, the reason. For unknown names:
        an error and the closest matching names.
    """
    key = _LOOKUP.get(short_name.strip().lower())
    if key is None:
        close = difflib.get_close_matches(short_name, list(_INDICES), n=5, cutoff=0.5)
        return json.dumps({
            "error": f"'{short_name}' is not in the Awesome Spectral Indices catalogue.",
            "did_you_mean": close,
            "hint": "Use search_spectral_indices to look up indices by topic.",
        })
    idx = _INDICES[key]
    bands, constants = [], {}
    for code in idx.bands:
        if code in spyndex.bands:
            band = spyndex.bands[code]
            bands.append({
                "code": code,
                "name": band.long_name,
                "sentinel2_band": S2_L2A_BANDS[code][0] if code in S2_L2A_BANDS else None,
            })
        elif code in spyndex.constants:
            c = spyndex.constants[code]
            constants[code] = {"description": c.long_name, "default": c.default}
    ok, reason = _S2_OK[key]
    out = {
        "short_name": idx.short_name,
        "long_name": idx.long_name,
        "application_domain": idx.application_domain,
        "formula": idx.formula,
        "bands": bands,
        "constants": constants or None,
        "reference": idx.reference,
        "platforms": list(idx.platforms),
        "sentinel2_computable": ok,
    }
    if not ok:
        out["reason"] = reason
    return json.dumps(out)


def catalogue_stats() -> dict:
    """Used by test.py: catalogue size and Sentinel-2 coverage per domain."""
    per_domain = Counter(i.application_domain for i in _INDICES.values())
    s2 = Counter(_INDICES[n].application_domain for n, (ok, _) in _S2_OK.items() if ok)
    return {"total": len(_INDICES), "sentinel2": sum(s2.values()), "per_domain": per_domain, "s2_per_domain": s2}



# ---------------------------------------------------------------------------
# Credentials (per-request headers first, then environment), as in servers/effis
# ---------------------------------------------------------------------------


def _request_headers() -> dict[str, str]:
    try:
        request = mcp.get_context().request_context.request
        if request is not None and hasattr(request, "headers"):
            return {k.lower(): v for k, v in dict(request.headers).items()}
    except Exception:
        pass
    return {}


def _cdse_credentials() -> tuple[str, str]:
    headers = _request_headers()
    cid = headers.get("x-cdse-client-id", "") or os.getenv("CDSE_CLIENT_ID", "")
    secret = headers.get("x-cdse-client-secret", "") or os.getenv("CDSE_CLIENT_SECRET", "")
    return cid, secret


_token_cache: dict[str, tuple[str, float]] = {}


async def _cdse_token(client_id: str, client_secret: str) -> str:
    cached = _token_cache.get(client_id)
    if cached and cached[1] > time.time() + 60:
        return cached[0]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    token = payload["access_token"]
    _token_cache[client_id] = (token, time.time() + float(payload.get("expires_in", 300)))
    return token


async def _process_tiff(token: str, body: dict[str, Any]) -> bytes:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = await client.post(
            CDSE_PROCESS_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/tiff",
            },
        )
    if resp.status_code != 200:
        detail = resp.text[:400]
        raise RuntimeError(f"CDSE Process API returned {resp.status_code}: {detail}")
    return resp.content


async def _statistics(token: str, body: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = await client.post(
            CDSE_STATS_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"CDSE Statistical API returned {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _decode_two_band_tiff(data: bytes, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (median_index, n_valid) arrays of shape (height, width)."""
    arr = tifffile.imread(io.BytesIO(data)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 2 and arr.shape[:2] == (height, width):
        median, count = arr[..., 0], arr[..., 1]
    elif arr.ndim == 3 and arr.shape[0] == 2:
        median, count = arr[0], arr[1]
    else:
        raise RuntimeError(f"Unexpected raster shape from CDSE: {arr.shape}")
    median = np.where(np.isfinite(median), median, np.nan)
    count = np.nan_to_num(count, nan=0.0)
    return median, count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(error_type: str, message: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error_type": error_type, "error": message, **extra})


def _resolve_index(index: str | None, formula: str | None) -> dict[str, Any]:
    """Return {short_name, long_name, formula, domain, reference} or raise KeyError."""
    if formula:
        return {
            "short_name": index or "CUSTOM",
            "long_name": "Custom formula",
            "formula": formula,
            "application_domain": None,
            "reference": None,
        }
    key = _LOOKUP.get((index or "").strip().lower())
    if key is None:
        raise KeyError(index)
    idx = spyndex.indices[key]
    return {
        "short_name": idx.short_name,
        "long_name": idx.long_name,
        "formula": idx.formula,
        "application_domain": idx.application_domain,
        "reference": idx.reference,
    }


def _prepare_index(index: str, formula: str | None) -> tuple[dict[str, Any], str, list[str], dict[str, float]] | str:
    """(info, js_expr, bands_used, constants_used), or an error JSON string."""
    try:
        info = _resolve_index(index, formula)
    except KeyError:
        return _error(
            "unknown_index",
            f"'{index}' is not an Awesome Spectral Indices name.",
            hint="Use search_spectral_indices to find one, e.g. NDVI, NDBI, MNDWI, NBR.",
        )
    if info["application_domain"] == "kernel":
        return _error("not_computable", f"{info['short_name']} is a kernel index and needs extra parameters.")
    try:
        js_expr, bands_used, consts_used = formula_to_js(info["formula"], "sentinel-2-l2a", _CONSTANTS)
    except ConversionError as exc:
        return _error("not_computable", f"{info['short_name']} cannot be computed from Sentinel-2: {exc}")
    return info, js_expr, bands_used, consts_used


async def _login() -> str:
    """CDSE access token; raises PermissionError with a user-facing message."""
    cid, secret = _cdse_credentials()  # resolved once, at the start of the tool call
    if not cid or not secret:
        raise PermissionError(
            "No CDSE credentials. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (or send "
            "X-CDSE-Client-Id / X-CDSE-Client-Secret headers)."
        )
    try:
        return await _cdse_token(cid, secret)
    except Exception as exc:
        raise PermissionError(f"Could not log in to CDSE: {exc}") from exc


def _index_block(info: dict[str, Any], consts_used: dict[str, float]) -> dict[str, Any]:
    return {
        "short_name": info["short_name"],
        "long_name": info["long_name"],
        "formula": info["formula"],
        "constants": consts_used or None,
        "reference": info["reference"],
    }


def _file_stem(name: str, start: Any, end: Any, box: tuple[float, ...]) -> str:
    tag = "_".join(f"{v:.3f}" for v in box).replace("-", "m")
    return f"{name}_{start}_{end}_{tag}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def compute_index_change(
    bbox: str,
    index: str,
    before_start: str,
    before_end: str,
    after_start: str,
    after_end: str,
    threshold: float = 0.1,
    method: str = "delta",
    max_cloud_cover: float = 40.0,
    resolution_m: float = 20.0,
    formula: str | None = None,
    save_map: bool = True,
) -> str:
    """Measure how a spectral index changed in an area between two periods (Sentinel-2).

    Use this for questions like "how much forest was lost", "how much built-up
    area was added", "did the lake grow" between two dates. Pick the index first
    (search_spectral_indices helps): NDVI for vegetation, NDBI for built-up,
    NDWI or MNDWI for water, NBR for burns, NDSI for snow. For "when did it
    change", call compute_index_timeseries afterwards with decrease_extent or
    increase_extent from this result as its bbox.

    For each period the tool takes the per-pixel median over all cloud-free
    Sentinel-2 L2A scenes (clouds, shadows, cirrus and snow removed), then
    delta = after - before. Pixels with delta > threshold count as increase,
    delta < -threshold as decrease.

    Choose the same season in both periods (e.g. June-August of each year), at
    least one month long, so seasonal greening is not counted as change.
    Sentinel-2 L2A data start in 2017.

    Args:
        bbox: "west,south,east,north" in WGS84 degrees, longitude first,
            e.g. "13.76,52.38,13.82,52.41". Max 2,500 km².
        index: Awesome Spectral Indices short name, e.g. "NDVI", "NDBI", "MNDWI".
        before_start: First day of the earlier period, YYYY-MM-DD.
        before_end: Last day of the earlier period, YYYY-MM-DD.
        after_start: First day of the later period, YYYY-MM-DD.
        after_end: Last day of the later period, YYYY-MM-DD.
        threshold: Minimum absolute index change that counts as change
            (default 0.1; for normalised indices 0.1-0.2 is typical).
        method: Change method. Only "delta" is available.
        max_cloud_cover: Skip scenes with more cloud than this, in % (default 40).
        resolution_m: Pixel size in metres (default 20; minimum 10).
        formula: Optional custom formula in ASI syntax using band codes
            B, G, R, RE1, RE2, RE3, N, N2, S1, S2 (e.g. "(S1 - N)/(S1 + N)").
            Overrides the catalogue formula of `index`.
        save_map: Save a PNG map: before, after, delta and change classes
            (orange increase, purple decrease) with hectares in the legend.

    Returns:
        JSON. On success ok=true with: index (name, formula, reference),
        area_ha, before / after mean, delta mean / median / p10 / p90,
        increase_ha, decrease_ha, stable_ha, valid_fraction, scenes per pixel,
        increase_extent / decrease_extent (bbox of the changed pixels, or null),
        interpretation, map_path (PNG to show the user), warnings. Report
        both increase_ha and decrease_ha. On failure ok=false with error_type
        (invalid_input, unknown_index, not_computable, cdse_auth, cdse_error,
        no_data) and error (a sentence to show the user).
    """
    warnings: list[str] = []

    if method not in METHODS:
        return _error("invalid_input", f"Unknown method '{method}'. Available: {', '.join(METHODS)}.")

    # ---- inputs ----
    try:
        box = parse_bbox(bbox)
        b0, b1 = parse_date(before_start, "before_start"), parse_date(before_end, "before_end")
        a0, a1 = parse_date(after_start, "after_start"), parse_date(after_end, "after_end")
    except ValueError as exc:
        return _error("invalid_input", str(exc))
    if b0 > b1 or a0 > a1:
        return _error("invalid_input", "Each period needs start <= end.")
    if b1 >= a0:
        return _error("invalid_input", "The 'before' period must end before the 'after' period starts.")
    if b0.year < 2017:
        warnings.append("Sentinel-2 L2A coverage before 2017 is sparse or missing; results may be empty.")
    if (b1 - b0).days < 20 or (a1 - a0).days < 20:
        warnings.append("A period shorter than ~3 weeks may have few cloud-free scenes.")
    month_gap = abs(b0.month - a0.month) % 12
    if min(month_gap, 12 - month_gap) > 1:
        warnings.append("The two periods are in different seasons; seasonal differences will look like change.")

    area_ha = bbox_area_ha(box)
    if area_ha > MAX_AREA_HA:
        return _error(
            "invalid_input",
            f"bbox covers {area_ha:,.0f} ha; the limit is {MAX_AREA_HA:,.0f} ha. Use a smaller box.",
        )

    prepared = _prepare_index(index, formula)
    if isinstance(prepared, str):
        return prepared
    info, js_expr, bands_used, consts_used = prepared

    width, height, res = grid_for_bbox(box, resolution_m)
    if res > resolution_m + 1e-6:
        warnings.append(f"Resolution coarsened to {res:.0f} m to stay within 2500 px.")
    pixel_ha = area_ha / (width * height)

    try:
        token = await _login()
    except PermissionError as exc:
        return _error("cdse_auth", str(exc))

    # ---- two median composites on the same grid ----
    evalscript = build_median_evalscript(js_expr, bands_used)
    bodies = [
        build_process_request(evalscript, box, str(b0), str(b1), width, height, max_cloud_cover),
        build_process_request(evalscript, box, str(a0), str(a1), width, height, max_cloud_cover),
    ]
    logger.info("compute_index_change %s bbox=%s grid=%dx%d @%.0fm", info["short_name"], box, width, height, res)
    try:
        tiffs = await asyncio.gather(*(_process_tiff(token, body) for body in bodies))
        before, n_before = _decode_two_band_tiff(tiffs[0], width, height)
        after, n_after = _decode_two_band_tiff(tiffs[1], width, height)
    except Exception as exc:
        return _error("cdse_error", f"Sentinel-2 data could not be retrieved: {exc}")

    stats = summarize_change(before, after, n_before, n_after, pixel_ha, threshold)
    if stats.get("empty"):
        return _error(
            "no_data",
            "No pixel had a cloud-free Sentinel-2 scene in both periods. "
            "Try longer periods or a higher max_cloud_cover.",
            scenes_per_pixel_before=stats["scenes_per_pixel_before"],
            scenes_per_pixel_after=stats["scenes_per_pixel_after"],
        )
    if stats["valid_fraction"] < 0.5:
        warnings.append(f"Only {stats['valid_fraction']:.0%} of pixels had cloud-free data in both periods.")

    delta = after.astype(np.float64) - before.astype(np.float64)
    with np.errstate(invalid="ignore"):
        stats["increase_extent"] = change_extent(delta > threshold, box)
        stats["decrease_extent"] = change_extent(delta < -threshold, box)

    map_path = None
    if save_map:
        try:
            DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            stem = _file_stem(info["short_name"], b0, a1, box)
            map_path = change_figure(
                before, after, threshold, info["short_name"], info["application_domain"],
                (str(b0), str(b1)), (str(a0), str(a1)), stats, box,
                DEFAULT_SAVE_DIR / f"{stem}_change.png",
            )
            (DEFAULT_SAVE_DIR / f"{stem}_before.tif").write_bytes(tiffs[0])
            (DEFAULT_SAVE_DIR / f"{stem}_after.tif").write_bytes(tiffs[1])
        except Exception as exc:
            warnings.append(f"Map not saved: {exc}")
            map_path = None

    result = {
        "ok": True,
        "method": method,
        "index": _index_block(info, consts_used),
        "data": "Sentinel-2 L2A (CDSE), per-pixel median of cloud-free scenes",
        "bbox": ",".join(f"{v:.5f}" for v in box),
        "area_ha": round(area_ha, 1),
        "resolution_m": round(res, 1),
        "before_period": [str(b0), str(b1)],
        "after_period": [str(a0), str(a1)],
        **stats,
        "interpretation": DOMAIN_HINT.get(info["application_domain"] or "", "Check the index definition for its direction."),
        "map_path": str(map_path) if map_path else None,
        "warnings": warnings,
    }
    return json.dumps(result)


@mcp.tool()
async def compute_index_timeseries(
    bbox: str,
    index: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 40.0,
    min_valid_fraction: float = 0.6,
    resolution_m: float = 20.0,
    formula: str | None = None,
    save_plot: bool = True,
) -> str:
    """Time series of a spectral index over an area, one value per Sentinel-2 scene.

    Use this for "how did NDVI develop here since 2018", "when was the forest
    cleared", "show the water index over the last years". For each Sentinel-2
    L2A acquisition day the tool computes the index on every pixel, removes
    clouds, shadows, cirrus and snow with the SCL layer, and returns the area
    mean and percentiles. Scenes where less than min_valid_fraction of the area
    is cloud-free are skipped. Results are grouped by month and year, and the
    change_point gives the month where the level of the series shifts most
    after removing the seasonal cycle (usually when a clearing, construction or
    flood happened). A plot (PNG) and the full table (CSV) are saved.

    The statistics are area averages over the whole bbox, so a change that
    covers only part of the box is diluted. To date a change found by
    compute_index_change, pass its decrease_extent (or increase_extent) as bbox
    here: the series then follows mainly the changed pixels.

    Args:
        bbox: "west,south,east,north" in WGS84 degrees, longitude first,
            e.g. "13.76,52.38,13.82,52.41", or decrease_extent / increase_extent
            from compute_index_change. Max 2,500 km².
        index: Awesome Spectral Indices short name, e.g. "NDVI", "NDBI", "MNDWI".
        start_date: First day, YYYY-MM-DD (Sentinel-2 L2A starts in 2017).
        end_date: Last day, YYYY-MM-DD. At most ten years after start_date.
        max_cloud_cover: Skip scenes with more cloud than this, in % (default 40).
        min_valid_fraction: Skip scenes where less than this share of the area
            is cloud-free (0-1, default 0.6).
        resolution_m: Pixel size in metres (default 20; use 60 for long or
            large requests to make them faster).
        formula: Optional custom formula in ASI syntax (band codes B, G, R,
            RE1, RE2, RE3, N, N2, S1, S2). Overrides the catalogue formula.
        save_plot: Save the PNG plot and CSV table.

    Returns:
        JSON. On success ok=true with: index (name, formula, reference),
        scenes_used, scenes_skipped, monthly ([month, median of scene means,
        n scenes]), yearly ({year: {median, n}}), trend_per_year,
        change_point ({month, shift, mean_before, mean_after}), min, max,
        plot_path (PNG to show the user), csv_path, warnings. On failure ok=false with
        error_type (invalid_input, unknown_index, not_computable, cdse_auth,
        cdse_error, no_data) and error.
    """
    warnings: list[str] = []
    try:
        box = parse_bbox(bbox)
        d0, d1 = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
    except ValueError as exc:
        return _error("invalid_input", str(exc))
    if d0 >= d1:
        return _error("invalid_input", "start_date must be before end_date.")
    if (d1 - d0).days > MAX_SERIES_DAYS:
        return _error("invalid_input", "The period can be at most ten years. Shorten it or split it.")
    if d0.year < 2017:
        warnings.append("Sentinel-2 L2A coverage before 2017 is sparse or missing.")
    if not 0 <= min_valid_fraction <= 1:
        return _error("invalid_input", "min_valid_fraction must be between 0 and 1.")
    area_ha = bbox_area_ha(box)
    if area_ha > MAX_AREA_HA:
        return _error("invalid_input", f"bbox covers {area_ha:,.0f} ha; the limit is {MAX_AREA_HA:,.0f} ha.")

    prepared = _prepare_index(index, formula)
    if isinstance(prepared, str):
        return prepared
    info, js_expr, bands_used, consts_used = prepared

    width, height, res = grid_for_bbox(box, resolution_m)
    if res > resolution_m + 1e-6:
        warnings.append(f"Resolution coarsened to {res:.0f} m to stay within 2500 px.")

    try:
        token = await _login()
    except PermissionError as exc:
        return _error("cdse_auth", str(exc))

    body = build_stats_request(
        build_stats_evalscript(js_expr, bands_used), box, str(d0), str(d1), width, height, max_cloud_cover
    )
    logger.info("compute_index_timeseries %s bbox=%s %s..%s grid=%dx%d", info["short_name"], box, d0, d1, width, height)
    try:
        payload = await _statistics(token, body)
    except Exception as exc:
        return _error("cdse_error", f"Sentinel-2 statistics could not be retrieved: {exc}")

    rows, skipped = parse_stats_response(payload, min_valid_fraction)
    if not rows:
        return _error(
            "no_data",
            "No Sentinel-2 scene had enough cloud-free pixels in this period. "
            "Try a longer period, a higher max_cloud_cover or a lower min_valid_fraction.",
            scenes_skipped=skipped,
        )
    summary = summarize_timeseries(rows)
    if len(rows) < 5:
        warnings.append(f"Only {len(rows)} usable scenes; the series is sparse.")

    plot_path = csv_path = None
    if save_plot:
        try:
            DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            stem = _file_stem(info["short_name"], d0, d1, box)
            csv_path = DEFAULT_SAVE_DIR / f"{stem}_timeseries.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            plot_path = timeseries_figure(
                rows, summary["monthly"], info["short_name"], box, DEFAULT_SAVE_DIR / f"{stem}_timeseries.png",
                summary.get("change_point"),
            )
            if plot_path is None:
                warnings.append("Plot not saved: matplotlib is not installed.")
        except Exception as exc:
            warnings.append(f"Files not saved: {exc}")

    result = {
        "ok": True,
        "index": _index_block(info, consts_used),
        "data": "Sentinel-2 L2A (CDSE), cloud-masked area statistics per acquisition day",
        "bbox": ",".join(f"{v:.5f}" for v in box),
        "area_ha": round(area_ha, 1),
        "resolution_m": round(res, 1),
        "period": [str(d0), str(d1)],
        "scenes_used": len(rows),
        "scenes_skipped": skipped,
        **summary,
        "interpretation": DOMAIN_HINT.get(info["application_domain"] or "", "Check the index definition for its direction."),
        "plot_path": str(plot_path) if plot_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "warnings": warnings,
    }
    return json.dumps(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectral Change MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port != mcp.settings.port:
        mcp.settings.port = args.port
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")

