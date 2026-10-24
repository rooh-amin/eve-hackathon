"""Awesome Spectral Indices (ASI) formulas -> Sentinel Hub evalscripts.

Pure Python + numpy, no network. Used by the change-detection server and,
as an identical copy, by the spectral-indices server.

* ``formula_to_js``   safely translates an ASI formula (Python syntax, as in
                      ``spyndex.indices[...].formula``) into a JS expression over
                      Sentinel Hub band names. Nothing is ``eval``-ed: the
                      formula is parsed with ``ast`` and only arithmetic on known
                      bands and constants is accepted.
* ``build_median_evalscript`` builds a VERSION=3 evalscript that computes the
                      index for every Sentinel-2 L2A scene in a window
                      (``mosaicking: ORBIT``), drops clouds/shadows/snow with the
                      SCL layer, and returns the per-pixel median plus the number
                      of valid scenes.
* ``build_process_request`` builds the Process API body for one window.
* ``summarize_change`` turns the before/after median rasters into a compact
                      JSON-ready summary (means, percentiles, hectares).
* ``change_extent``   the bounding box of the changed pixels, so a time series
                      can be focused on where the change happened.
* ``build_stats_evalscript`` / ``build_stats_request`` / ``parse_stats_response``
                      / ``summarize_timeseries``: per-scene index statistics over
                      an area from the Statistical API (cloud-masked), grouped
                      into months and years.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Band mapping
# ---------------------------------------------------------------------------

# ASI band code -> (Sentinel Hub band name, centre wavelength nm (S2A), native resolution m)
S2_L2A_BANDS: dict[str, tuple[str, float, int]] = {
    "A": ("B01", 442.7, 60),
    "B": ("B02", 492.4, 10),
    "G": ("B03", 559.8, 10),
    "R": ("B04", 664.6, 10),
    "RE1": ("B05", 704.1, 20),
    "RE2": ("B06", 740.5, 20),
    "RE3": ("B07", 782.8, 20),
    "N": ("B08", 832.8, 10),
    "N2": ("B8A", 864.7, 20),
    "WV": ("B09", 945.1, 60),
    "S1": ("B11", 1613.7, 20),
    "S2": ("B12", 2202.4, 20),
}

PLATFORMS: dict[str, dict[str, tuple[str, float, int]]] = {
    "sentinel-2-l2a": S2_L2A_BANDS,
}

# ASI wavelength constants (lambdaN, lambdaR, ...) resolved for Sentinel-2.
S2_LAMBDA: dict[str, float] = {f"lambda{code}": wl for code, (_, wl, _) in S2_L2A_BANDS.items()}

# Scene Classification Layer classes treated as invalid:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob.,
# 9 cloud high prob., 10 thin cirrus, 11 snow/ice.
S2_BAD_SCL: tuple[int, ...] = (0, 1, 3, 8, 9, 10, 11)

# Only used when allow_funcs=True (custom formulas). The ASI catalogue itself
# contains no function calls.
_FUNCS = {
    "sqrt": "Math.sqrt",
    "log": "Math.log",
    "log10": "Math.log10",
    "exp": "Math.exp",
    "abs": "Math.abs",
}

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

EARTH_RADIUS_M = 6_371_008.8
PROCESS_MAX_PX = 2500


class ConversionError(ValueError):
    """The formula cannot be computed on this platform (or is not allowed)."""


# ---------------------------------------------------------------------------
# Formula translation
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    text = repr(float(value))
    if text in ("inf", "-inf", "nan"):
        raise ConversionError(f"Non-finite constant {value!r} is not allowed.")
    return f"({text})" if value < 0 else text


def formula_to_js(
    formula: str,
    platform: str = "sentinel-2-l2a",
    constants: dict[str, float | None] | None = None,
    allow_funcs: bool = False,
) -> tuple[str, list[str], dict[str, float]]:
    """Translate an ASI formula into a JS expression over ``s.<band>``.

    Args:
        formula: ASI formula, e.g. ``"(N - R)/(N + R)"``.
        platform: Key of ``PLATFORMS``.
        constants: Values for ASI constants (``L``, ``g``, ``C1``...). ``None``
            values count as unknown. Wavelength constants (``lambdaN``...) are
            filled in from the platform automatically.
        allow_funcs: Allow ``sqrt``, ``log``, ``log10``, ``exp``, ``abs``.

    Returns:
        ``(js_expression, sentinel_hub_bands_used, constants_used)``.

    Raises:
        ConversionError: unknown name, band missing on the platform, or an
            unsupported construct.
    """
    if platform not in PLATFORMS:
        raise ConversionError(f"Unknown platform '{platform}'. Known: {sorted(PLATFORMS)}")
    bands = PLATFORMS[platform]
    lambdas = S2_LAMBDA if platform == "sentinel-2-l2a" else {}
    consts = {k: v for k, v in (constants or {}).items() if v is not None}

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise ConversionError(f"Formula is not valid syntax: {exc.msg}") from exc

    used_bands: set[str] = set()
    used_consts: dict[str, float] = {}

    def walk(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                return f"Math.pow({left}, {right})"
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise ConversionError(f"Operator {type(node.op).__name__} is not allowed.")
            return f"({left} {op} {right})"
        if isinstance(node, ast.UnaryOp):
            inner = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{inner})"
            if isinstance(node.op, ast.UAdd):
                return f"(+{inner})"
            raise ConversionError(f"Unary {type(node.op).__name__} is not allowed.")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ConversionError(f"Literal {node.value!r} is not a number.")
            return _num(node.value)
        if isinstance(node, ast.Name):
            name = node.id
            if name in bands:
                sh_band = bands[name][0]
                used_bands.add(sh_band)
                return f"s.{sh_band}"
            if name in lambdas:
                used_consts[name] = lambdas[name]
                return _num(lambdas[name])
            if name in consts:
                used_consts[name] = float(consts[name])
                return _num(consts[name])
            raise ConversionError(
                f"'{name}' is not a {platform} band or a known constant "
                f"(available bands: {', '.join(bands)})."
            )
        if isinstance(node, ast.Call):
            if not allow_funcs:
                raise ConversionError("Function calls are not allowed in catalogue formulas.")
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
                raise ConversionError(f"Only these functions are allowed: {', '.join(_FUNCS)}.")
            args = ", ".join(walk(a) for a in node.args)
            return f"{_FUNCS[node.func.id]}({args})"
        raise ConversionError(f"Unsupported expression: {type(node).__name__}.")

    js = walk(tree)
    if not used_bands:
        raise ConversionError("Formula uses no spectral bands.")
    return js, sorted(used_bands), used_consts


def asi_bands_to_sentinel2(bands: list[str]) -> dict[str, str]:
    """Map ASI band codes to Sentinel-2 band names (only those that exist)."""
    return {b: S2_L2A_BANDS[b][0] for b in bands if b in S2_L2A_BANDS}


# ---------------------------------------------------------------------------
# Evalscript + request
# ---------------------------------------------------------------------------


def build_median_evalscript(js_expr: str, bands_used: list[str]) -> str:
    """Per-pixel median of the index over all valid scenes in the time range.

    Output (FLOAT32, 2 bands): [median_index, n_valid_scenes]. Pixels with no
    valid scene return [NaN, 0].
    """
    inputs = ", ".join(f'"{b}"' for b in [*bands_used, "SCL", "dataMask"])
    bad = ", ".join(str(c) for c in S2_BAD_SCL)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{inputs}] }}],
    output: {{ id: "default", bands: 2, sampleType: "FLOAT32" }},
    mosaicking: "ORBIT"
  }};
}}
var BAD_SCL = [{bad}];
function indexValue(s) {{
  return {js_expr};
}}
function median(v) {{
  v.sort(function (a, b) {{ return a - b; }});
  var m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}}
function evaluatePixel(samples) {{
  var vals = [];
  for (var i = 0; i < samples.length; i++) {{
    var s = samples[i];
    if (s.dataMask !== 1) continue;
    if (BAD_SCL.indexOf(s.SCL) !== -1) continue;
    var v = indexValue(s);
    if (isFinite(v)) vals.push(v);
  }}
  if (vals.length === 0) return [NaN, 0];
  return [median(vals), vals.length];
}}
"""


def build_process_request(
    evalscript: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    width: int,
    height: int,
    max_cloud_cover: float,
) -> dict[str, Any]:
    """Process API body for one Sentinel-2 L2A window (FLOAT32 GeoTIFF out)."""
    return {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                        "maxCloudCoverage": float(max_cloud_cover),
                    },
                    # Removes the +1000 DN offset of processing baseline 04.00+
                    # (Jan 2022) so 2019 and 2024 scenes are comparable.
                    "processing": {"harmonizeValues": True},
                }
            ],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def parse_bbox(bbox: str | list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    """Accept ``"west,south,east,north"`` or ``[west, south, east, north]``."""
    if isinstance(bbox, str):
        parts = [p for p in bbox.replace(";", ",").replace(" ", ",").split(",") if p]
    else:
        parts = list(bbox)
    if len(parts) != 4:
        raise ValueError("bbox needs 4 numbers: west,south,east,north (WGS84 degrees).")
    try:
        west, south, east, north = (float(p) for p in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numbers.") from exc
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("bbox values must be finite.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(
            "bbox must be west,south,east,north with west<east and south<north "
            "(longitude first, WGS84 degrees)."
        )
    return west, south, east, north


def bbox_area_ha(bbox: str | list[float] | tuple[float, ...]) -> float:
    """Area of a lon/lat box on a sphere, in hectares."""
    west, south, east, north = parse_bbox(bbox)
    area_m2 = (
        EARTH_RADIUS_M**2
        * math.radians(east - west)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )
    return abs(area_m2) / 10_000.0


def grid_for_bbox(
    bbox: tuple[float, float, float, float], resolution_m: float
) -> tuple[int, int, float]:
    """Pixel grid (width, height, effective resolution m) for a box, capped at 2500 px."""
    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2)
    width_m = EARTH_RADIUS_M * math.cos(mid_lat) * math.radians(east - west)
    height_m = EARTH_RADIUS_M * math.radians(north - south)
    res = max(float(resolution_m), 10.0)
    res = max(res, width_m / PROCESS_MAX_PX, height_m / PROCESS_MAX_PX)
    width = max(1, round(width_m / res))
    height = max(1, round(height_m / res))
    return width, height, res


def parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a date in YYYY-MM-DD format, got {value!r}.") from exc


# ---------------------------------------------------------------------------
# Change summary
# ---------------------------------------------------------------------------


def _r(value: float, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def summarize_change(
    before: np.ndarray,
    after: np.ndarray,
    n_before: np.ndarray,
    n_after: np.ndarray,
    pixel_ha: float,
    threshold: float,
) -> dict[str, Any]:
    """Compact statistics of ``delta = after - before`` (per-pixel medians)."""
    before = before.astype(np.float64)
    after = after.astype(np.float64)
    delta = after - before
    valid = np.isfinite(before) & np.isfinite(after)
    total = int(valid.size)
    n_valid = int(valid.sum())
    out: dict[str, Any] = {
        "pixels_total": total,
        "pixels_valid": n_valid,
        "valid_fraction": _r(n_valid / total if total else 0.0, 3),
        "scenes_per_pixel_before": _r(float(np.median(n_before[n_before > 0])) if np.any(n_before > 0) else 0.0, 1),
        "scenes_per_pixel_after": _r(float(np.median(n_after[n_after > 0])) if np.any(n_after > 0) else 0.0, 1),
    }
    if n_valid == 0:
        out["empty"] = True
        return out
    d = delta[valid]
    b = before[valid]
    a = after[valid]
    inc = d > threshold
    dec = d < -threshold
    p10, p50, p90 = np.percentile(d, [10, 50, 90])
    out.update(
        {
            "before_mean": _r(b.mean()),
            "after_mean": _r(a.mean()),
            "delta_mean": _r(d.mean()),
            "delta_median": _r(p50),
            "delta_p10": _r(p10),
            "delta_p90": _r(p90),
            "threshold": threshold,
            "increase_ha": _r(inc.sum() * pixel_ha, 1),
            "decrease_ha": _r(dec.sum() * pixel_ha, 1),
            "stable_ha": _r((~inc & ~dec).sum() * pixel_ha, 1),
            "increase_pct_of_valid": _r(100.0 * inc.mean(), 2),
            "decrease_pct_of_valid": _r(100.0 * dec.mean(), 2),
            # Mean change inside the pixels that crossed the threshold.
            "delta_mean_where_increase": _r(d[inc].mean()) if inc.any() else None,
            "delta_mean_where_decrease": _r(d[dec].mean()) if dec.any() else None,
        }
    )
    return out


def change_extent(
    changed: np.ndarray,
    bbox: tuple[float, float, float, float],
    min_pixels: int = 10,
    pad_px: int = 1,
) -> str | None:
    """Bounding box ("west,south,east,north") of the changed pixels.

    ``changed`` is a boolean (height, width) mask on the request grid (row 0 =
    north). The 2nd-98th percentile of the changed rows/columns is used, so a few
    isolated noisy pixels far away do not blow up the box. Returns None when
    fewer than ``min_pixels`` pixels changed.
    """
    rows, cols = np.nonzero(changed)
    if rows.size < min_pixels:
        return None
    h, w = changed.shape
    r0, r1 = np.percentile(rows, [2, 98])
    c0, c1 = np.percentile(cols, [2, 98])
    r0, c0 = max(int(np.floor(r0)) - pad_px, 0), max(int(np.floor(c0)) - pad_px, 0)
    r1, c1 = min(int(np.ceil(r1)) + pad_px, h - 1), min(int(np.ceil(c1)) + pad_px, w - 1)
    west, south, east, north = bbox
    dx, dy = (east - west) / w, (north - south) / h
    return f"{west + c0 * dx:.5f},{north - (r1 + 1) * dy:.5f},{west + (c1 + 1) * dx:.5f},{north - r0 * dy:.5f}"


# ---------------------------------------------------------------------------
# Time series (Statistical API)
# ---------------------------------------------------------------------------


def build_stats_evalscript(js_expr: str, bands_used: list[str]) -> str:
    """One index value per pixel and scene; clouds, shadows, cirrus and snow masked out.

    Outputs: "index" (FLOAT32) and "dataMask" (pixels with dataMask 0 are left
    out of the statistics by the Statistical API).
    """
    inputs = ", ".join(f'"{b}"' for b in [*bands_used, "SCL", "dataMask"])
    bad = ", ".join(str(c) for c in S2_BAD_SCL)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{inputs}] }}],
    output: [
      {{ id: "index", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
var BAD_SCL = [{bad}];
function evaluatePixel(s) {{
  var ok = s.dataMask === 1 && BAD_SCL.indexOf(s.SCL) === -1;
  var v = ok ? ({js_expr}) : 0;
  if (!isFinite(v)) {{ ok = false; v = 0; }}
  return {{ index: [v], dataMask: [ok ? 1 : 0] }};
}}
"""


def build_stats_request(
    evalscript: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    width: int,
    height: int,
    max_cloud_cover: float,
    interval: str = "P1D",
) -> dict[str, Any]:
    """Statistical API body: one statistics entry per acquisition day (P1D)."""
    return {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"maxCloudCoverage": float(max_cloud_cover)},
                    "processing": {"harmonizeValues": True},
                }
            ],
        },
        "aggregation": {
            "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
            "aggregationInterval": {"of": interval},
            "width": int(width),
            "height": int(height),
            "evalscript": evalscript,
        },
        "calculations": {"index": {"statistics": {"default": {"percentiles": {"k": [10, 50, 90]}}}}},
    }


def parse_stats_response(payload: dict[str, Any], min_valid_fraction: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rows (one per usable scene date) and counts of skipped entries."""
    rows: list[dict[str, Any]] = []
    skipped = {"cloudy_or_partial": 0, "errors": 0}
    for entry in payload.get("data", []):
        if entry.get("error"):
            skipped["errors"] += 1
            continue
        try:
            stats = entry["outputs"]["index"]["bands"]["B0"]["stats"]
            day = entry["interval"]["from"][:10]
        except (KeyError, TypeError):
            skipped["errors"] += 1
            continue
        total = float(stats.get("sampleCount") or 0)
        nodata = float(stats.get("noDataCount") or 0)
        valid = (total - nodata) / total if total else 0.0
        mean = stats.get("mean")
        if valid < min_valid_fraction or mean is None or not math.isfinite(float(mean)):
            skipped["cloudy_or_partial"] += 1
            continue
        pct = stats.get("percentiles") or {}
        rows.append(
            {
                "date": day,
                "mean": float(mean),
                "p10": _f(pct.get("10.0")),
                "median": _f(pct.get("50.0")),
                "p90": _f(pct.get("90.0")),
                "valid_fraction": round(valid, 3),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows, skipped


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summarize_timeseries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Monthly and yearly medians of the scene means, trend, and the main change point.

    * monthly: [["YYYY-MM", median of scene means, n scenes], ...]
    * yearly: {"YYYY": {"median": .., "n": ..}}
    * trend_per_year: least-squares slope through the monthly values
    * change_point: the month where the level of the series shifts most. The
      seasonal cycle is removed first (each calendar month minus its mean over
      all years), then the split that best separates "before" from "after" is
      chosen (largest |mean_after - mean_before| weighted by sample sizes).
      Usually this is when a clearing, construction or flood happened.
    """
    if not rows:
        return {"monthly": [], "yearly": {}}
    by_month: dict[str, list[float]] = defaultdict(list)
    by_year: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_month[r["date"][:7]].append(r["mean"])
        by_year[r["date"][:4]].append(r["mean"])
    monthly = [[m, round(float(np.median(v)), 4), len(v)] for m, v in sorted(by_month.items())]
    yearly = {y: {"median": round(float(np.median(v)), 4), "n": len(v)} for y, v in sorted(by_year.items())}

    out: dict[str, Any] = {"monthly": monthly, "yearly": yearly}
    months = [m for m, _, _ in monthly]
    y = np.array([v for _, v, _ in monthly], dtype=float)
    if len(monthly) >= 3:
        t = np.array([int(m[:4]) + (int(m[5:7]) - 0.5) / 12 for m in months])
        out["trend_per_year"] = round(float(np.polyfit(t, y, 1)[0]), 4)
    if len(monthly) >= 6:
        resid = y.copy()
        if len({m[:4] for m in months}) >= 2:
            cal = np.array([int(m[5:7]) for m in months])
            for c in set(cal.tolist()):
                resid[cal == c] -= y[cal == c].mean()
        n = len(resid)
        best = None
        for k in range(2, n - 1):
            shift = resid[k:].mean() - resid[:k].mean()
            score = abs(shift) * math.sqrt(k * (n - k) / n)
            if best is None or score > best[0]:
                best = (score, k, shift)
        if best is not None:
            _, k, shift = best
            out["change_point"] = {
                "month": months[k],
                "shift": round(float(shift), 4),
                "mean_before": round(float(y[:k].mean()), 4),
                "mean_after": round(float(y[k:].mean()), 4),
            }
    values = [(r["mean"], r["date"]) for r in rows]
    out["min"] = {"date": min(values)[1], "value": round(min(values)[0], 4)}
    out["max"] = {"date": max(values)[1], "value": round(max(values)[0], 4)}
    return out
