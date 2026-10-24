# spectral-change

MCP server: which spectral index fits a question, how much it changed in an area between two periods, and when it changed.

- **Indices:** the [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices) catalogue via `spyndex` (280 indices, 239 computable from Sentinel-2), or a custom formula in the same syntax.
- **Data:** Sentinel-2 L2A on the Copernicus Data Space Ecosystem (CDSE), through the Sentinel Hub Process and Statistical APIs.

## Tools

| Tool | What it does | Needs CDSE |
|---|---|---|
| `search_spectral_indices(query, application_domain, sentinel2_only, limit)` | Find indices by topic: "built-up" → NDBI, "flooding" → MNDWI, "forest loss" → NDVI. | no |
| `get_spectral_index(short_name)` | Formula, bands with Sentinel-2 band names, constants, reference, and whether Sentinel-2 can compute it. | no |
| `compute_index_change(bbox, index, before_start, before_end, after_start, after_end, threshold=0.1, method="delta", ...)` | Per-pixel median before vs after, delta, hectares of increase / decrease, extent of the changed pixels, 4-panel map. | yes |
| `compute_index_timeseries(bbox, index, start_date, end_date, ...)` | Cloud-masked area statistics for every scene, monthly / yearly medians, trend, change point, plot + CSV. | yes |

Typical chain: `search_spectral_indices` → `compute_index_change` (how much) → `compute_index_timeseries` with the change tool's `decrease_extent` or `increase_extent` as bbox (when). The extent keeps the time series focused on the changed pixels instead of averaging them with the unchanged rest of the box.

## How it works

1. `asi_formula.py` turns the spyndex formula (e.g. `(N-R)/(N+R)`) into a Sentinel Hub evalscript. It parses with Python's `ast` and accepts only arithmetic on Sentinel-2 bands and catalogue constants; nothing is `eval`-ed.
2. **CDSE computes the index**, scene by scene, with clouds, shadows, cirrus and snow removed via the SCL layer. `harmonizeValues` is on, so scenes before and after the January 2022 processing change are comparable.
   - **Change:** one Process API request per period returns the per-pixel median (`mosaicking: ORBIT`) as a small FLOAT32 GeoTIFF.
   - **Time series:** one Statistical API request (`P1D`) returns area statistics per acquisition day. Scenes with less than `min_valid_fraction` cloud-free area are skipped.
3. This server computes delta, hectares, extents, monthly and yearly summaries and the change point with numpy. It draws the figures with matplotlib (`plots.py`) and writes PNG / CSV / GeoTIFF files to `CHANGE_MAPS_DIR` (default `<tmp>/spectral_change`).
4. Tools return compact JSON with the numbers and file paths. Errors return `ok: false` with `error_type` (`invalid_input`, `unknown_index`, `not_computable`, `cdse_auth`, `cdse_error`, `no_data`) and a sentence for the user.

## Credentials

A CDSE OAuth client ([create one](https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings)). The server reads the headers `X-CDSE-Client-Id` / `X-CDSE-Client-Secret` first, then `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` from the environment or from a `.env` in this folder or up to three parent folders. For a registry deployment, the two headers must be added to the allowlist in `scripts/deploy_server.py`.

## Run and test

```bash
pip install -r requirements.txt
python server.py --transport stdio        # or --transport http  -> http://localhost:8000/mcp
python test.py                            # offline; evalscripts run in node against a numpy reference
python test.py --live                     # plus real CDSE calls (Tesla Gigafactory, Grünheide)
```

Possible next methods (same tool, `method=`): built-up as NDVI drop AND NDBI rise, per-pixel trend, change vector analysis on several indices.
