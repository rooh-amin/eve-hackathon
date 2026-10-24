"""Tests for the spectral-change server.

    python test.py          # offline: formulas, evalscript (needs node), stats, tool with mocked CDSE
    python test.py --live   # also real CDSE calls (Tesla Gigafactory, Grünheide); needs CDSE_* in .env

The evalscript is run in Node.js and compared with a numpy reference, so the
JavaScript that Sentinel Hub executes is tested, not only the Python around it.
Steps that need node are skipped when node is not on PATH.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import spyndex
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asi_formula import (  # noqa: E402
    PLATFORMS,
    S2_BAD_SCL,
    S2_LAMBDA,
    ConversionError,
    bbox_area_ha,
    build_median_evalscript,
    build_stats_evalscript,
    change_extent,
    formula_to_js,
    grid_for_bbox,
    parse_bbox,
    parse_stats_response,
    summarize_change,
    summarize_timeseries,
)

RNG = np.random.default_rng(7)
CONSTS = {k: c.default for k, c in spyndex.constants.items()}
CODE_OF = {v[0]: code for code, v in PLATFORMS["sentinel-2-l2a"].items()}
HAS_NODE = shutil.which("node") is not None
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def py_eval(formula: str, values: dict[str, np.ndarray]) -> np.ndarray:
    ns = {**{k: v for k, v in CONSTS.items() if v is not None}, **S2_LAMBDA, **values}
    with np.errstate(all="ignore"):
        return np.asarray(eval(compile(formula, "<f>", "eval"), {"__builtins__": {}}, ns), dtype=float)  # noqa: S307


def same(a, b, rtol=1e-9, atol=1e-12) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    fin = np.isfinite(a) & np.isfinite(b)
    return bool(
        (np.isnan(a) == np.isnan(b)).all()
        and (np.isinf(a) == np.isinf(b)).all()
        and np.allclose(a[fin], b[fin], rtol=rtol, atol=atol)
    )


def run_node(source: str, payload: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(source)
    r = subprocess.run(["node", f.name], input=json.dumps(payload), capture_output=True, text=True, check=False)
    Path(f.name).unlink(missing_ok=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


# --------------------------------------------------------------------- 0 catalogue tools
def call(fn, **kwargs) -> dict:
    return json.loads(asyncio.run(fn(**kwargs)))


def test_catalogue_tools() -> None:
    import server

    stats = server.catalogue_stats()
    check("catalogue loaded", stats["total"] >= 250, f"{stats['total']} indices, {stats['sentinel2']} on Sentinel-2")
    missing = (server.POPULAR | server.CANONICAL) - set(server._INDICES)
    check("popular names exist", not missing, str(missing))

    r = call(server.search_spectral_indices, query="built-up area")
    names = [x["short_name"] for x in r["results"]]
    check("'built-up area' -> urban indices, NDBI first", names and names[0] == "NDBI" and r["domains_matched"] == ["urban"],
          ", ".join(names[:5]))
    r = call(server.search_spectral_indices, query="flooding")
    names = [x["short_name"] for x in r["results"]]
    check("'flooding' -> water indices incl. MNDWI", "MNDWI" in names, ", ".join(names[:5]))
    r = call(server.search_spectral_indices, query="forest loss")
    check("'forest loss' -> NDVI in top 3", "NDVI" in [x["short_name"] for x in r["results"][:3]])
    r = call(server.search_spectral_indices, query="NDVI")
    check("exact name ranks first", r["results"][0]["short_name"] == "NDVI")
    r = call(server.search_spectral_indices, query="", application_domain="burn", limit=50)
    check("domain listing, Sentinel-2 only", r["total_matches"] > 5 and all(x["sentinel2"] for x in r["results"]),
          f"{r['total_matches']} burn indices")
    r = call(server.search_spectral_indices, query="thermal", sentinel2_only=False, application_domain="urban")
    check("sentinel2_only=False includes thermal indices", any(not x["sentinel2"] for x in r["results"]))
    r = call(server.search_spectral_indices, query="xyzzy")
    check("no match gives a hint", r["total_matches"] == 0 and "hint" in r)
    r = call(server.search_spectral_indices, query="water", application_domain="oceans")
    check("unknown domain lists valid domains", "error" in r and "water" in r["domains"])
    r = call(server.get_spectral_index, short_name="evi")
    s2 = {b["code"]: b["sentinel2_band"] for b in r["bands"]}
    check("get EVI: bands mapped, constants with defaults",
          s2 == {"N": "B08", "R": "B04", "B": "B02"} and r["constants"]["g"]["default"] == 2.5 and r["sentinel2_computable"],
          json.dumps(s2))
    r = call(server.get_spectral_index, short_name="NDBaI")
    check("thermal index not computable, with reason", r["sentinel2_computable"] is False and "T" in r["reason"])
    r = call(server.get_spectral_index, short_name="NDVII")
    check("unknown name suggests close matches", "NDVI" in r.get("did_you_mean", []), str(r.get("did_you_mean")))
    sizes = [len(asyncio.run(server.search_spectral_indices(query=q))) for q in ("vegetation", "water", "urban")]
    check("search outputs stay compact (< 3 kB)", max(sizes) < 3000, f"{max(sizes)} bytes")
    tools = set(getattr(server.mcp, "tools", {}) or {})
    if tools:  # only visible with the local test shim
        check("server exposes the 4 tools", tools == {"search_spectral_indices", "get_spectral_index",
                                                       "compute_index_change", "compute_index_timeseries"}, str(tools))


def test_change_extent() -> None:
    mask = np.zeros((100, 200), bool)
    mask[20:40, 50:90] = True  # rows 20-39 from the north, cols 50-89 from the west
    mask[95, 5] = True  # one isolated noisy pixel must not stretch the box
    ext = change_extent(mask, (10.0, 50.0, 12.0, 51.0))
    w, s, e, n = (float(v) for v in ext.split(","))
    check("change_extent: box around the block, noise ignored",
          abs(w - 10.49) < 0.02 and abs(e - 10.91) < 0.02 and abs(n - 50.81) < 0.02 and abs(s - 50.59) < 0.02, ext)
    check("change_extent: None for < 10 changed pixels", change_extent(np.zeros((10, 10), bool), (0, 0, 1, 1)) is None)


# --------------------------------------------------------------------- 1 formulas
def test_catalogue_formulas(n: int = 2000) -> None:
    computable = {}
    for name, idx in spyndex.indices.items():
        if idx.application_domain == "kernel":
            continue
        try:
            js, bands, _ = formula_to_js(idx.formula, "sentinel-2-l2a", CONSTS)
            computable[name] = (idx.formula, js, bands)
        except ConversionError:
            pass
    check("catalogue: >= 230 indices computable on Sentinel-2", len(computable) >= 230, f"{len(computable)}")
    if not HAS_NODE:
        print("SKIP  catalogue JS == Python (node not found)")
        return
    sh_bands = [v[0] for v in PLATFORMS["sentinel-2-l2a"].values()]
    data = {b: RNG.uniform(0.0, 0.6, n) for b in sh_bands}
    for b in data:
        data[b][:3] = 0.0  # division by zero paths
    src = "const d = JSON.parse(require('fs').readFileSync(0,'utf8')); const out = {};\n"
    for name, (_, js, _) in computable.items():
        src += (
            f"out[{json.dumps(name)}] = Array.from({{length: d.n}}, (_, i) => {{ const s = {{}};"
            f" for (const k in d.b) s[k] = d.b[k][i]; const v = {js};"
            " return Number.isFinite(v) ? v : (Number.isNaN(v) ? 'nan' : (v > 0 ? 'inf' : '-inf')); });\n"
        )
    src += "process.stdout.write(JSON.stringify(out));\n"
    got = run_node(src, {"n": n, "b": {k: v.tolist() for k, v in data.items()}})
    bad = []
    for name, (formula, _, _) in computable.items():
        ref = py_eval(formula, {CODE_OF[b]: v for b, v in data.items()})
        ref = np.broadcast_to(ref, (n,))
        js_vals = np.array([float(x) for x in got[name]])
        if not same(js_vals, ref, rtol=1e-7):
            bad.append(name)
    check("catalogue: JS translation == Python on 2000 random pixels", not bad, f"mismatch: {bad[:8]}" if bad else "")


# --------------------------------------------------------------------- 2 rejections
def test_rejections() -> None:
    cases = [
        "__import__('os').system('x')",
        "N.real",
        "(N - T) / (N + T)",
        "(VV - VH) / (VV + VH)",
        "N if R else G",
        "[N, R]",
        "N // R",
        "sqrt(N)",
        "2 + 3",
        "(N - R) / (N + R) + unknown_constant",
    ]
    failures = []
    for formula in cases:
        try:
            formula_to_js(formula, "sentinel-2-l2a", CONSTS)
            failures.append(formula)
        except ConversionError:
            pass
    check("unsafe or non-Sentinel-2 formulas are rejected", not failures, str(failures))
    js, _, _ = formula_to_js("sqrt(N) - log10(R)", "sentinel-2-l2a", allow_funcs=True)
    check("functions allowed only on request", js == "(Math.sqrt(s.B08) - Math.log10(s.B04))", js)


# --------------------------------------------------------------------- 3 evalscript
HARNESS = r"""
const vm = require('vm');
const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const ctx = {}; vm.createContext(ctx); vm.runInContext(d.script, ctx);
const cfg = ctx.setup();
const res = d.pixels.map(p => ctx.evaluatePixel(p));
process.stdout.write(JSON.stringify({cfg, res}, (k, v) => (typeof v === 'number' && !isFinite(v)) ? 'nan' : v));
"""


def test_evalscript(index: str = "NDBI", n_pix: int = 300, n_scenes: int = 9) -> None:
    if not HAS_NODE:
        print("SKIP  evalscript median logic (node not found)")
        return
    formula = spyndex.indices[index].formula
    js, bands, _ = formula_to_js(formula, "sentinel-2-l2a", CONSTS)
    script = build_median_evalscript(js, bands)
    pixels, ref = [], []
    scl_choices = [4, 5, 6, 7, 2, 0, 1, 3, 8, 9, 10, 11]
    scl_p = [0.3, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    for _ in range(n_pix):
        samples, vals = [], []
        for _ in range(int(RNG.integers(0, n_scenes + 1))):
            s = {b: float(RNG.uniform(0.01, 0.6)) for b in bands}
            s["dataMask"] = int(RNG.random() > 0.1)
            s["SCL"] = int(RNG.choice(scl_choices, p=scl_p))
            samples.append(s)
            if s["dataMask"] == 1 and s["SCL"] not in S2_BAD_SCL:
                v = float(py_eval(formula, {CODE_OF[b]: np.array(s[b]) for b in bands}))
                if np.isfinite(v):
                    vals.append(v)
        pixels.append(samples)
        ref.append([float(np.median(vals)), len(vals)] if vals else [np.nan, 0])
    got = run_node(HARNESS, {"script": script, "pixels": pixels})
    res = [[np.nan if x == "nan" else x for x in r] for r in got["res"]]
    check(f"evalscript ({index}): per-pixel cloud-masked median == numpy", same(res, ref))
    cfg = got["cfg"]
    check(
        "evalscript setup: ORBIT mosaicking, FLOAT32, 2 bands, SCL + dataMask inputs",
        cfg["mosaicking"] == "ORBIT"
        and cfg["output"]["sampleType"] == "FLOAT32"
        and cfg["output"]["bands"] == 2
        and {"SCL", "dataMask"} <= set(cfg["input"][0]["bands"]),
    )


# --------------------------------------------------------------------- 4 geometry + stats
def test_geometry_and_stats() -> None:
    check("1x1 degree at the equator ~ 1.236 M ha", abs(bbox_area_ha("0,0,1,1") - 1_236_372) < 2_000)
    w, h, res = grid_for_bbox(parse_bbox("13.76,52.38,13.82,52.41"), 20)
    check("Grünheide box at 20 m ~ 204 x 167 px", (w, h, res) == (204, 167, 20.0), f"{w}x{h}@{res}")
    w, h, res = grid_for_bbox(parse_bbox("10,50,12,52"), 10)
    check("large boxes are capped at 2500 px", max(w, h) <= 2500 and res > 10, f"{w}x{h}@{res:.1f}")
    for bad in ["13.82,52.38,13.76,52.41", "52.38,13.76,52.41", "a,b,c,d", "200,0,201,1"]:
        try:
            parse_bbox(bad)
            check(f"bbox rejected: {bad}", False)
        except ValueError:
            pass
    before = np.full((10, 10), 0.2)
    after = before.copy()
    after[:3, :] = 0.5  # 30 px increase
    after[9, :] = -0.1  # 10 px decrease
    before[0, 0] = np.nan  # 1 px no data
    s = summarize_change(before, after, np.full((10, 10), 5), np.full((10, 10), 4), pixel_ha=0.04, threshold=0.1)
    check(
        "summary: hectares and valid fraction",
        s["increase_ha"] == round(29 * 0.04, 1) and s["decrease_ha"] == 0.4 and s["valid_fraction"] == 0.99,
        json.dumps({k: s[k] for k in ("increase_ha", "decrease_ha", "valid_fraction")}),
    )


# --------------------------------------------------------------------- 5 tool with mocked CDSE
def _fake_scene_values(bands, built: bool, rng) -> dict[str, float]:
    """Pine forest before; a factory roof/concrete block after."""
    forest = {"B02": 0.03, "B03": 0.05, "B04": 0.03, "B05": 0.08, "B06": 0.2, "B07": 0.25,
              "B08": 0.3, "B8A": 0.3, "B09": 0.3, "B11": 0.15, "B12": 0.07, "B01": 0.03}
    concrete = {"B02": 0.12, "B03": 0.14, "B04": 0.16, "B05": 0.18, "B06": 0.19, "B07": 0.2,
                "B08": 0.2, "B8A": 0.21, "B09": 0.2, "B11": 0.3, "B12": 0.25, "B01": 0.12}
    base = concrete if built else forest
    return {b: base[b] * float(rng.uniform(0.95, 1.05)) for b in bands}


def make_mock_process(changed_block: tuple[slice, slice]):
    """A stand-in for CDSE: runs the real evalscript in node on synthetic scenes."""

    async def fake_process(token: str, body: dict) -> bytes:
        script = body["evalscript"]
        w, h = body["output"]["width"], body["output"]["height"]
        start = body["input"]["data"][0]["dataFilter"]["timeRange"]["from"]
        after_period = start >= "2023"
        bands = re.search(r"bands: \[(.*?)\]", script).group(1).replace('"', "").split(", ")
        bands = [b for b in bands if b not in ("SCL", "dataMask")]
        rng = np.random.default_rng(1)
        built = np.zeros((h, w), bool)
        built[changed_block] = after_period
        pixels = []
        for y in range(h):
            for x in range(w):
                scenes = []
                for k in range(6):
                    s = _fake_scene_values(bands, built[y, x], rng)
                    s["dataMask"] = 1
                    s["SCL"] = 9 if k == 0 else 4  # first scene cloudy everywhere
                    scenes.append(s)
                pixels.append(scenes)
        got = run_node(HARNESS, {"script": script, "pixels": pixels})
        arr = np.array([[np.nan if v == "nan" else v for v in r] for r in got["res"]], dtype=np.float32)
        buf = io.BytesIO()
        tifffile.imwrite(buf, arr.reshape(h, w, 2))
        return buf.getvalue()

    return fake_process


@contextlib.contextmanager
def mocked_cdse(server, process=None, statistics=None):
    """Swap the CDSE calls for fakes and ALWAYS restore them (so --live hits real CDSE)."""
    saved = {k: getattr(server, k) for k in ("_cdse_token", "_cdse_credentials", "_process_tiff", "_statistics", "DEFAULT_SAVE_DIR")}

    async def fake_token(cid, secret):
        return "token"

    server._cdse_token = fake_token
    server._cdse_credentials = lambda: ("id", "secret")
    if process:
        server._process_tiff = process
    if statistics:
        server._statistics = statistics
    server.DEFAULT_SAVE_DIR = Path(tempfile.mkdtemp())
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(server, k, v)


def test_tool_mocked() -> None:
    if not HAS_NODE:
        print("SKIP  tool with mocked CDSE (node not found)")
        return
    import server

    block = (slice(10, 30), slice(10, 40))  # 20 x 30 px
    with mocked_cdse(server, process=make_mock_process(block)):
        args = dict(bbox="13.76,52.38,13.82,52.41", before_start="2019-06-01", before_end="2019-08-31",
                    after_start="2024-06-01", after_end="2024-08-31", resolution_m=60)
        ndbi = json.loads(asyncio.run(server.compute_index_change(index="ndbi", **args)))
        w, h, _ = grid_for_bbox(parse_bbox(args["bbox"]), 60)
        expected_ha = 20 * 30 * bbox_area_ha(args["bbox"]) / (w * h)
        check("tool: ok + NDBI increase_ha matches the changed block", ndbi.get("ok") and
              abs(ndbi["increase_ha"] - expected_ha) < 1 and ndbi["decrease_ha"] == 0,
              f"{ndbi.get('increase_ha')} vs {expected_ha:.1f} ha")
        check("tool: change map PNG written", ndbi.get("map_path") and Path(ndbi["map_path"]).read_bytes()[:4] == b"\x89PNG",
              str(ndbi.get("map_path")))
        check("tool: output is compact (< 2 kB)", len(json.dumps(ndbi)) < 2000, f"{len(json.dumps(ndbi))} bytes")
        ndvi = json.loads(asyncio.run(server.compute_index_change(index="NDVI", **args)))
        check("tool: NDVI shows the same block as decrease", ndvi.get("ok") and abs(ndvi["decrease_ha"] - expected_ha) < 1,
              f"{ndvi.get('decrease_ha')} ha")
        ext = ndvi.get("decrease_extent")
        box = parse_bbox(args["bbox"])
        exp_w = box[0] + 10 * (box[2] - box[0]) / w
        exp_n = box[3] - 10 * (box[3] - box[1]) / h
        ok_ext = ext is not None and abs(float(ext.split(",")[0]) - exp_w) < 0.003 and abs(float(ext.split(",")[3]) - exp_n) < 0.003
        check("tool: decrease_extent outlines the changed block; increase_extent is null for NDVI",
              ok_ext and ndvi.get("increase_extent") is None, str(ext))

        errors = {
            "unknown_index": dict(args, index="NOPE"),
            "not_computable": dict(args, index="LST" if "LST" in spyndex.indices else "NDBaI"),
            "invalid_input": dict(args, index="NDVI", before_end="2025-01-01"),
        }
        for expected, kwargs in errors.items():
            out = json.loads(asyncio.run(server.compute_index_change(**kwargs)))
            check(f"tool error: {expected}", out.get("ok") is False and out.get("error_type") in (expected, "not_computable"),
                  out.get("error", "")[:90])
    check("mocks restored after the mocked tests", server._process_tiff.__name__ == "_process_tiff"
          and server._statistics.__name__ == "_statistics")


# --------------------------------------------------------------------- 6 time series
def test_stats_evalscript(index: str = "NDVI", n_pix: int = 400) -> None:
    if not HAS_NODE:
        print("SKIP  statistics evalscript (node not found)")
        return
    formula = spyndex.indices[index].formula
    js, bands, _ = formula_to_js(formula, "sentinel-2-l2a", CONSTS)
    script = build_stats_evalscript(js, bands)
    pixels, ref = [], []
    for i in range(n_pix):
        s = {b: float(RNG.uniform(0.0, 0.6)) for b in bands}
        if i < 3:
            s = {b: 0.0 for b in bands}  # 0/0 -> must be masked, not NaN in the stats
        s["dataMask"] = int(RNG.random() > 0.1)
        s["SCL"] = int(RNG.choice([4, 5, 6, 3, 8, 9, 10, 11, 0, 1]))
        pixels.append(s)
        ok = s["dataMask"] == 1 and s["SCL"] not in S2_BAD_SCL
        v = float(py_eval(formula, {CODE_OF[b]: np.array(s[b]) for b in bands})) if ok else 0.0
        if not np.isfinite(v):
            ok, v = False, 0.0
        ref.append([v, 1 if ok else 0])
    got = run_node(HARNESS, {"script": script, "pixels": pixels})
    res = [[r["index"][0], r["dataMask"][0]] for r in got["res"]]
    outs = [o["id"] for o in got["cfg"]["output"]]
    check(f"statistics evalscript ({index}): SCL/NaN masking == numpy", same(res, ref) and outs == ["index", "dataMask"])


def fake_stats_payload(start: str, end: str, change_day: str = "2020-02-15") -> dict:
    """Synthetic Statistical API answer: seasonal NDVI that drops by 0.35 on change_day."""
    from datetime import date, timedelta

    rng = np.random.default_rng(3)
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    data = []
    while d <= last:
        season = 0.08 * np.sin(2 * np.pi * (d.timetuple().tm_yday - 100) / 365)
        level = 0.75 if d.isoformat() < change_day else 0.40
        mean = level + season + rng.normal(0, 0.015)
        cloudy = rng.random() < 0.3
        entry = {"interval": {"from": f"{d}T00:00:00Z", "to": f"{d + timedelta(days=1)}T00:00:00Z"},
                 "outputs": {"index": {"bands": {"B0": {"stats": {
                     "min": mean - 0.3, "max": mean + 0.2, "mean": mean, "stDev": 0.1,
                     "sampleCount": 1000, "noDataCount": 900 if cloudy else 20,
                     "percentiles": {"10.0": mean - 0.1, "50.0": mean, "90.0": mean + 0.08}}}}}}}
        data.append(entry)
        d += timedelta(days=5)
    data.append({"interval": {"from": "2021-01-01T00:00:00Z"}, "error": {"type": "EXECUTION_ERROR"}})
    return {"data": data, "status": "OK"}


def test_timeseries_parsing() -> None:
    rows, skipped = parse_stats_response(fake_stats_payload("2018-01-01", "2024-12-31"), min_valid_fraction=0.6)
    check("parse: cloudy scenes and errors skipped", skipped["errors"] == 1 and skipped["cloudy_or_partial"] > 50
          and all(r["valid_fraction"] >= 0.6 for r in rows), json.dumps(skipped))
    summ = summarize_timeseries(rows)
    cp = summ.get("change_point", {})
    check("summary: change point found at the clearing (2020-02/03)", cp.get("month") in ("2020-02", "2020-03"),
          json.dumps(cp))
    check("summary: shift ~ -0.35", abs(cp.get("shift", 0) + 0.35) < 0.05, str(cp.get("shift")))
    check("summary: yearly 2019 > 2021", summ["yearly"]["2019"]["median"] > summ["yearly"]["2021"]["median"] + 0.25)
    check("summary: trend is negative", summ["trend_per_year"] < 0, str(summ["trend_per_year"]))


def test_timeseries_tool_mocked() -> None:
    import server

    seen = {}

    async def fake_statistics(token, body):
        seen["body"] = body
        tr = body["aggregation"]["timeRange"]
        return fake_stats_payload(tr["from"][:10], tr["to"][:10])

    with mocked_cdse(server, statistics=fake_statistics):
        out = json.loads(asyncio.run(server.compute_index_timeseries(
            bbox="13.76,52.38,13.82,52.41", index="NDVI", start_date="2018-01-01", end_date="2024-12-31")))
        body = seen.get("body", {})
        check("timeseries: request is P1D with harmonizeValues and the dataMask output",
              body.get("aggregation", {}).get("aggregationInterval", {}).get("of") == "P1D"
              and body["input"]["data"][0]["processing"]["harmonizeValues"] is True
              and 'id: "dataMask"' in body["aggregation"]["evalscript"])
        check("timeseries: ok, change point 2020-02/03", out.get("ok") and out["change_point"]["month"] in ("2020-02", "2020-03"),
              json.dumps(out.get("change_point")))
        plot = Path(out.get("plot_path") or "/nonexistent")
        check("timeseries: plot PNG + CSV written", plot.exists() and plot.read_bytes()[:4] == b"\x89PNG"
              and Path(out["csv_path"]).read_text().startswith("date,mean"), str(plot))
        size = len(json.dumps(out))
        check("timeseries: output fits an agent context (< 6 kB for 7 years)", size < 6000, f"{size} bytes")
        bad = json.loads(asyncio.run(server.compute_index_timeseries(
            bbox="13.76,52.38,13.82,52.41", index="NDVI", start_date="2010-01-01", end_date="2024-12-31")))
        check("timeseries error: > 10 years rejected", bad.get("error_type") == "invalid_input", bad.get("error", ""))


# --------------------------------------------------------------------- 7 live
def test_live() -> None:
    import server

    out = json.loads(asyncio.run(server.compute_index_change(
        bbox="13.76,52.38,13.82,52.41", index="NDBI",
        before_start="2019-06-01", before_end="2019-08-31",
        after_start="2024-06-01", after_end="2024-08-31",
    )))
    print(json.dumps(out, indent=2))
    check("live change (Grünheide, NDBI 2019 vs 2024)", out.get("ok") is True and out["scenes_per_pixel_before"] != 5.0,
          out.get("error", out.get("map_path", "")))
    ts = json.loads(asyncio.run(server.compute_index_timeseries(
        bbox="13.76,52.38,13.82,52.41", index="NDVI", start_date="2018-01-01", end_date="2024-12-31",
    )))
    short = {k: v for k, v in ts.items() if k != "monthly"}
    print(json.dumps(short, indent=2))
    check("live time series (Grünheide, NDVI 2018-2024)", ts.get("ok") is True, ts.get("error", ts.get("plot_path", "")))

    ndvi = json.loads(asyncio.run(server.compute_index_change(
        bbox="13.76,52.38,13.82,52.41", index="NDVI",
        before_start="2019-06-01", before_end="2019-08-31",
        after_start="2024-06-01", after_end="2024-08-31",
    )))
    ext = ndvi.get("decrease_extent")
    print("NDVI decrease_ha:", ndvi.get("decrease_ha"), "| decrease_extent:", ext)
    if ext:
        focus = json.loads(asyncio.run(server.compute_index_timeseries(
            bbox=ext, index="NDVI", start_date="2018-01-01", end_date="2024-12-31",
        )))
        print(json.dumps({k: focus.get(k) for k in ("ok", "bbox", "scenes_used", "change_point", "yearly", "plot_path")}, indent=2))
        check("live focused time series on decrease_extent", focus.get("ok") is True, focus.get("error", ""))


if __name__ == "__main__":
    test_catalogue_tools()
    test_change_extent()
    test_catalogue_formulas()
    test_rejections()
    test_evalscript("NDBI")
    test_evalscript("EVI")
    test_geometry_and_stats()
    test_tool_mocked()
    test_stats_evalscript("NDVI")
    test_stats_evalscript("EVI")
    test_timeseries_parsing()
    test_timeseries_tool_mocked()
    if "--live" in sys.argv:
        test_live()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)
