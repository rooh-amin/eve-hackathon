"""Figures for the change-detection server (matplotlib, headless).

* ``change_figure``     2 x 2 panel: before median, after median, delta, change classes
* ``timeseries_figure`` scene means with p10-p90 range, plus the monthly median line

If matplotlib is missing, ``change_figure`` falls back to a plain classified PNG
written with zlib, and ``timeseries_figure`` returns None.
"""

from __future__ import annotations

import struct
import zlib
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    HAVE_MPL = True
except Exception:  # pragma: no cover - only without matplotlib
    HAVE_MPL = False

INCREASE = "#e6550d"  # orange
DECREASE = "#5e3c99"  # purple
STABLE = "#d9d9d9"

_INDEX_CMAP = {"vegetation": "YlGn", "water": "Blues", "urban": "magma", "burn": "inferno", "snow": "Blues_r"}


def _write_png(path: Path, rgba: np.ndarray) -> None:
    h, _, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", rgba.shape[1], h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _classes(delta: np.ndarray, threshold: float) -> np.ndarray:
    """0 no data, 1 stable, 2 increase, 3 decrease."""
    out = np.zeros(delta.shape, dtype=np.uint8)
    valid = np.isfinite(delta)
    out[valid] = 1
    out[valid & (delta > threshold)] = 2
    out[valid & (delta < -threshold)] = 3
    return out


def change_figure(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float,
    index_name: str,
    domain: str | None,
    before_period: tuple[str, str],
    after_period: tuple[str, str],
    stats: dict[str, Any],
    bbox: tuple[float, float, float, float],
    path: Path,
) -> Path:
    delta = after.astype(np.float64) - before.astype(np.float64)
    classes = _classes(delta, threshold)
    if not HAVE_MPL:
        colors = np.array([[0, 0, 0, 0], [217, 217, 217, 255], [230, 85, 13, 255], [94, 60, 153, 255]], np.uint8)
        _write_png(path, colors[classes])
        return path

    west, south, east, north = bbox
    extent = (west, east, south, north)
    both = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    vmin, vmax = (np.percentile(both, [2, 98]) if both.size else (0.0, 1.0))
    finite_delta = delta[np.isfinite(delta)]
    dlim = max(np.percentile(np.abs(finite_delta), 98) if finite_delta.size else 0.0, 2 * threshold)
    cmap = _INDEX_CMAP.get(domain or "", "viridis")
    aspect = 1 / np.cos(np.radians((south + north) / 2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)
    panels = [
        (axes[0, 0], before, f"{index_name} before\n{before_period[0]} – {before_period[1]}", cmap, vmin, vmax),
        (axes[0, 1], after, f"{index_name} after\n{after_period[0]} – {after_period[1]}", cmap, vmin, vmax),
        (axes[1, 0], delta, f"Δ {index_name} (after − before)", "PuOr_r", -dlim, dlim),
    ]
    for ax, data, title, cm, lo, hi in panels:
        im = ax.imshow(data, cmap=cm, vmin=lo, vmax=hi, extent=extent, aspect=aspect, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    class_cmap = ListedColormap(["#ffffff00", STABLE, INCREASE, DECREASE])
    ax.imshow(classes, cmap=class_cmap, vmin=0, vmax=3, extent=extent, aspect=aspect, interpolation="nearest")
    ax.set_title(f"Change classes (|Δ| > {threshold:g})", fontsize=11)
    ax.legend(
        handles=[
            Patch(color=INCREASE, label=f"increase  {stats.get('increase_ha', 0):,.1f} ha"),
            Patch(color=DECREASE, label=f"decrease  {stats.get('decrease_ha', 0):,.1f} ha"),
            Patch(color=STABLE, label=f"stable  {stats.get('stable_ha', 0):,.1f} ha"),
        ],
        loc="lower left",
        fontsize=9,
        framealpha=0.9,
    )
    for a in axes.flat:
        a.tick_params(labelsize=8)
        a.set_xlabel("lon", fontsize=8)
        a.set_ylabel("lat", fontsize=8)
    fig.suptitle(
        f"{index_name} change, Sentinel-2 L2A (CDSE) · bbox {west:.4f},{south:.4f},{east:.4f},{north:.4f}",
        fontsize=12,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def timeseries_figure(
    rows: list[dict[str, Any]],
    monthly: list[list[Any]],
    index_name: str,
    bbox: tuple[float, float, float, float],
    path: Path,
    change_point: dict[str, Any] | None = None,
) -> Path | None:
    if not HAVE_MPL or not rows:
        return None
    days = [date.fromisoformat(r["date"]) for r in rows]
    means = np.array([r["mean"] for r in rows])
    lo = np.array([r["p10"] if r["p10"] is not None else r["mean"] for r in rows])
    hi = np.array([r["p90"] if r["p90"] is not None else r["mean"] for r in rows])
    months = [date.fromisoformat(m + "-15") for m, _, _ in monthly]
    month_vals = [v for _, v, _ in monthly]

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.vlines(days, lo, hi, color="#9ecae1", linewidth=1, alpha=0.7, label="scene p10–p90")
    ax.scatter(days, means, s=12, color="#3182bd", zorder=3, label="scene mean (cloud-masked)")
    ax.plot(months, month_vals, color="#08306b", linewidth=1.8, zorder=4, label="monthly median")
    if change_point:
        cp = date.fromisoformat(change_point["month"] + "-01")
        ax.axvline(cp, color=INCREASE, linestyle="--", linewidth=1.5, zorder=5,
                   label=f"change point {change_point['month']} (shift {change_point['shift']:+.2f})")
    west, south, east, north = bbox
    ax.set_title(
        f"{index_name} time series, Sentinel-2 L2A (CDSE) · bbox {west:.4f},{south:.4f},{east:.4f},{north:.4f}",
        fontsize=11,
    )
    ax.set_ylabel(index_name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
