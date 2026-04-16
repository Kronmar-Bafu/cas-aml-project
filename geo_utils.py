"""
Geospatial utilities for the GNN-SDM pipeline.

Contains shared configuration, DEM processing, slope/aspect calculation,
and a generic CHELSA bioclimatic variable processor.
"""

import gc
import os

import numpy as np
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import boto3
import rasterio
import s3fs

# ---------------------------------------------------------------------------
# Shared spatial configuration (CHELSAch extent over Switzerland)
# ---------------------------------------------------------------------------

BOUNDS = {
    "lat_min": 45.56927314539999,
    "lat_max": 48.07159314539999,
    "lon_min": 5.7157257961,
    "lon_max": 10.6627057961,
}


def apply_map_decor(ax, title):
    """Apply consistent cartopy map decorations (borders, lakes, extent)."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=1)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.4, color="blue")
    ax.set_extent(
        [BOUNDS["lon_min"], BOUNDS["lon_max"], BOUNDS["lat_min"], BOUNDS["lat_max"]]
    )
    ax.set_title(title)


# ---------------------------------------------------------------------------
# DEM helpers
# ---------------------------------------------------------------------------


def get_tile_list(bounds=None):
    """Return S3 paths for all Copernicus GLO-30 DEM tiles covering *bounds*."""
    b = bounds or BOUNDS
    tiles = []
    for lat in range(int(np.floor(b["lat_min"])), int(np.ceil(b["lat_max"]))):
        for lon in range(int(np.floor(b["lon_min"])), int(np.ceil(b["lon_max"]))):
            tile = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            tiles.append(f"s3://copernicus-dem-30m/{tile}/{tile}.tif")
    return tiles


def load_dem(bounds=None):
    """Load, merge and clip the Copernicus GLO-30 DEM for the given bounds."""
    b = bounds or BOUNDS
    tile_paths = get_tile_list(b)
    print(f"Loading {len(tile_paths)} tiles from S3...")

    session = boto3.Session()
    aws_session = rasterio.session.AWSSession(session)

    with rasterio.Env(aws_session):
        datasets = [
            rioxarray.open_rasterio(t, chunks={"x": 1024, "y": 1024})
            for t in tile_paths
        ]
        merged = merge_arrays(datasets)
        clipped = merged.rio.clip_box(
            minx=b["lon_min"], miny=b["lat_min"],
            maxx=b["lon_max"], maxy=b["lat_max"],
        )
    return clipped


# ---------------------------------------------------------------------------
# Slope & aspect
# ---------------------------------------------------------------------------


def calculate_slope_aspect(elevation, res=30.0):
    """
    Calculate slope (degrees) and aspect (degrees, 0=N clockwise) from an
    elevation array.

    Parameters
    ----------
    elevation : 2-D numpy array
    res : pixel resolution in metres (default 30 m for Copernicus GLO-30)

    Returns
    -------
    slope_deg, aspect_deg : tuple of 2-D numpy arrays
    """
    dy, dx = np.gradient(elevation, res)

    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    aspect_deg = np.degrees(np.arctan2(-dy, dx))
    aspect_deg = (270.0 - aspect_deg) % 360.0

    return slope_deg, aspect_deg


def dem_to_slope_aspect(full_dem):
    """Convenience wrapper: DEM DataArray → (slope_da, aspect_da)."""
    slope_raw, aspect_raw = calculate_slope_aspect(full_dem.values[0], res=30.0)

    coords = full_dem.sel(band=1).coords
    dims = full_dem.sel(band=1).dims

    slope_da = xr.DataArray(slope_raw, coords=coords, dims=dims, name="slope")
    slope_da = slope_da.rio.write_crs("EPSG:4326")

    aspect_da = xr.DataArray(aspect_raw, coords=coords, dims=dims, name="aspect")
    aspect_da = aspect_da.rio.write_crs("EPSG:4326")

    return slope_da, aspect_da


def calculate_twi(elevation, transform, crs="EPSG:4326"):
    """
    Topographic Wetness Index with proper D8 flow accumulation.

    Uses ``pysheds`` for pit-filling, D8 flow direction, and flow
    accumulation so that the specific catchment area *a* reflects
    the true upslope contributing area.

    TWI = ln(a / tan(β))

    Parameters
    ----------
    elevation : 2-D numpy array
    transform : affine.Affine  (from rioxarray: ``da.rio.transform()``)
    crs : str or CRS

    Returns
    -------
    twi : 2-D numpy array (float32)
    """
    # pysheds 0.5 uses np.in1d which was removed in numpy 2.x
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder

    vf = ViewFinder(
        affine=transform,
        shape=elevation.shape,
        crs=str(crs),
        nodata=np.nan,
    )
    grid = Grid(viewfinder=vf)
    dem_raster = Raster(elevation.astype(np.float64), viewfinder=vf)

    # Fill pits, depressions, resolve flats
    print("  TWI: filling pits...")
    pit_filled = grid.fill_pits(dem_raster)
    print("  TWI: filling depressions...")
    flooded = grid.fill_depressions(pit_filled)
    print("  TWI: resolving flats...")
    inflated = grid.resolve_flats(flooded)

    # D8 flow direction and accumulation
    print("  TWI: computing flow direction...")
    fdir = grid.flowdir(inflated)
    print("  TWI: computing flow accumulation...")
    acc = np.asarray(grid.accumulation(fdir), dtype=np.float64)

    # Specific catchment area per unit contour length
    cell_deg = abs(transform.a)
    lat_rad = np.radians(46.8)  # mid-latitude for Switzerland
    cell_m = cell_deg * 111_320 * np.cos(lat_rad)
    a = (acc + 1) * cell_m

    # Slope in radians
    dy, dx = np.gradient(elevation, cell_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_rad = np.clip(slope_rad, 1e-6, None)

    twi = np.log(a / np.tan(slope_rad)).astype(np.float32)
    print("  TWI: done.")
    return twi


def calculate_curvature(elevation, res=30.0):
    """
    Plan and profile curvature from a DEM.

    - Profile curvature: curvature in the direction of steepest descent
      (positive = convex upslope / accelerating flow, negative = concave).
    - Plan curvature: curvature perpendicular to slope direction
      (positive = divergent flow, negative = convergent).

    Uses second-order finite differences via ``np.gradient``.

    Returns
    -------
    profile_curv, plan_curv : tuple of 2-D numpy arrays
    """
    dy, dx = np.gradient(elevation, res)
    dyy, dyx = np.gradient(dy, res)
    dxy, dxx = np.gradient(dx, res)

    p = dx**2 + dy**2
    q = p + 1.0
    # Avoid division by zero in flat areas
    p_safe = np.clip(p, 1e-10, None)

    profile_curv = -(dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / (
        p_safe * np.sqrt(q**3)
    )
    plan_curv = -(dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) / (
        p_safe * np.sqrt(q)
    )

    return profile_curv, plan_curv


def dem_to_terrain_features(full_dem):
    """
    Derive all terrain features from a DEM DataArray.

    Returns
    -------
    dict of xarray.DataArray with keys:
        slope, aspect, twi, profile_curvature, plan_curvature
    """
    elev = full_dem.values[0]
    coords = full_dem.sel(band=1).coords
    dims = full_dem.sel(band=1).dims
    crs = full_dem.rio.crs

    def _wrap(data, name):
        da = xr.DataArray(data, coords=coords, dims=dims, name=name)
        return da.rio.write_crs(crs)

    slope, aspect = calculate_slope_aspect(elev)
    twi = calculate_twi(elev, full_dem.rio.transform(), crs=str(crs))
    prof_curv, plan_curv = calculate_curvature(elev)

    return {
        "slope": _wrap(slope, "slope"),
        "aspect": _wrap(aspect, "aspect"),
        "twi": _wrap(twi, "twi"),
        "profile_curvature": _wrap(prof_curv, "profile_curvature"),
        "plan_curvature": _wrap(plan_curv, "plan_curvature"),
    }


def compute_forest_edge_distance(lc_fractions, threshold=0.5):
    """
    Compute distance-to-forest-edge from the tree-cover fraction layer.

    Pixels with tree-cover fraction >= *threshold* are considered forest.
    The result is a signed distance field in metres: positive outside
    forest, negative inside, zero at the edge.

    Parameters
    ----------
    lc_fractions : xarray.DataArray with a "band" dim containing "Tree cover"
    threshold : float

    Returns
    -------
    xarray.DataArray (y, x) — signed distance in metres
    """
    from scipy.ndimage import distance_transform_edt

    tree = lc_fractions.sel(band="Tree cover")
    forest_mask = (tree.values >= threshold).squeeze()

    # Pixel spacing in metres (approximate at Swiss mid-latitude)
    transform = tree.rio.transform() if tree.rio.transform() else lc_fractions.rio.transform()
    deg_per_px = abs(transform.a)
    lat_rad = np.radians(46.8)
    m_per_px_x = deg_per_px * 111_320 * np.cos(lat_rad)
    m_per_px_y = deg_per_px * 111_320
    sampling = (m_per_px_y, m_per_px_x)  # (row, col) order

    # Distance from non-forest to nearest forest pixel
    dist_outside = distance_transform_edt(~forest_mask, sampling=sampling).astype(np.float32)
    # Distance from forest to nearest non-forest pixel
    dist_inside = distance_transform_edt(forest_mask, sampling=sampling).astype(np.float32)

    # Signed: positive outside, negative inside
    signed_dist = dist_outside - dist_inside

    coords = tree.squeeze("band").coords if "band" in tree.dims else tree.coords
    dims = [d for d in tree.dims if d != "band"]

    da = xr.DataArray(signed_dist, coords=coords, dims=dims, name="forest_edge_distance")
    if tree.rio.crs:
        da = da.rio.write_crs(tree.rio.crs)
    return da


# ---------------------------------------------------------------------------
# Generic CHELSA bioclimatic variable processor
# ---------------------------------------------------------------------------


def process_chelsa_variable(
    target_grid,
    s3: s3fs.S3FileSystem,
    base_path: str,
    s3_zarr_path: str,
    *,
    var_prefix: str,       # "tas" or "pr"
    aggregation: str,      # "mean", "sum", or "std"
    dataset_name: str,     # name used in the output xr.Dataset
    months: list[int] | None = None,  # defaults to 1..12
):
    """
    Generic processor for CHELSA monthly data → single bioclimatic layer.

    Loads monthly NetCDF files from S3, applies *aggregation* over time,
    regrids to *target_grid* (bilinear), and writes the result as a
    chunked Zarr store.

    Parameters
    ----------
    var_prefix : 'tas' or 'pr'
    aggregation : 'mean' | 'sum' | 'std'
    dataset_name : variable name in the output Dataset
    months : which months to include (default all 12)
    """
    months = months or list(range(1, 13))
    files = [
        f"{base_path}/CHELSA_{var_prefix}_{m:02d}_1981-2010_V.1.0.nc"
        for m in months
    ]

    print(f"--- Processing {dataset_name} ({aggregation} of {var_prefix}) ---")

    try:
        monthly_data = []
        for f_path in files:
            print(f"  Loading {os.path.basename(f_path)}...")
            with s3.open(f_path, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf").load()
                da = ds[list(ds.data_vars)[0]]
                if "time" not in da.dims:
                    da = da.expand_dims("time")
                monthly_data.append(da)

        stack = xr.concat(monthly_data, dim="time")

        print(f"  Aggregating ({aggregation})...")
        if aggregation == "mean":
            result = stack.mean(dim="time")
        elif aggregation == "sum":
            result = stack.sum(dim="time")
        elif aggregation == "std":
            result = stack.std(dim="time")
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        print("  Regridding to 30 m master grid...")
        result = result.rio.write_crs("EPSG:4326").rio.reproject_match(
            target_grid, resampling=Resampling.bilinear
        )

        out_ds = result.to_dataset(name=dataset_name).chunk({"x": 512, "y": 512})

        print(f"  Writing Zarr → {s3_zarr_path}")
        store = s3fs.S3Map(root=s3_zarr_path, s3=s3, check=False)
        out_ds.to_zarr(store=store, mode="w", consolidated=True)
        print("  Done.")

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        gc.collect()


def process_bio15(target_grid, s3, base_path, s3_zarr_path):
    """
    Bio15: Precipitation seasonality (coefficient of variation).

    This one has custom logic (CV = std/mean * 100), so it stays separate.
    """
    print("--- Processing bio15_precip_seasonality (CV of pr) ---")
    files = [
        f"{base_path}/CHELSA_pr_{m:02d}_1981-2010_V.1.0.nc" for m in range(1, 13)
    ]

    try:
        monthly_data = []
        for f_path in files:
            print(f"  Loading {os.path.basename(f_path)}...")
            with s3.open(f_path, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf").load()
                da = ds[list(ds.data_vars)[0]]
                if "time" not in da.dims:
                    da = da.expand_dims("time")
                monthly_data.append(da)

        stack = xr.concat(monthly_data, dim="time")
        bio15 = (stack.std(dim="time") / (stack.mean(dim="time") + 0.001)) * 100

        print("  Regridding to 30 m master grid...")
        bio15 = bio15.rio.write_crs("EPSG:4326").rio.reproject_match(
            target_grid, resampling=Resampling.bilinear
        )

        out_ds = bio15.to_dataset(name="bio15_precip_seasonality").chunk(
            {"x": 512, "y": 512}
        )
        store = s3fs.S3Map(root=s3_zarr_path, s3=s3, check=False)
        out_ds.to_zarr(store=store, mode="w", consolidated=True)
        print("  Done.")

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        gc.collect()


def process_bio18(target_grid, s3, base_path, s3_zarr_path):
    """
    Bio18: Precipitation of the warmest quarter.

    Identifies the 3 warmest months globally, then sums their precipitation.
    """
    print("--- Processing bio18_warmest_quarter_precip ---")

    try:
        temp_means = []
        for m in range(1, 13):
            path = f"{base_path}/CHELSA_tas_{m:02d}_1981-2010_V.1.0.nc"
            with s3.open(path, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf").load()
                temp_means.append(float(ds[list(ds.data_vars)[0]].mean()))

        warmest_months = (np.argsort(temp_means)[-3:] + 1).tolist()
        print(f"  Warmest months: {warmest_months}")

        precip_layers = []
        for m in warmest_months:
            path = f"{base_path}/CHELSA_pr_{m:02d}_1981-2010_V.1.0.nc"
            print(f"  Loading precip month {m}...")
            with s3.open(path, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf").load()
                precip_layers.append(ds[list(ds.data_vars)[0]])

        bio18 = xr.concat(precip_layers, dim="time").sum(dim="time")

        print("  Regridding to 30 m master grid...")
        bio18 = bio18.rio.write_crs("EPSG:4326").rio.reproject_match(
            target_grid, resampling=Resampling.bilinear
        )

        out_ds = bio18.to_dataset(name="bio18_warmest_quarter_precip").chunk(
            {"x": 512, "y": 512}
        )
        store = s3fs.S3Map(root=s3_zarr_path, s3=s3, check=False)
        out_ds.to_zarr(store=store, mode="w", consolidated=True)
        print("  Done.")

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# ESA Worldcover (10 m → 30 m fraction-based regridding)
# ---------------------------------------------------------------------------

# ESA Worldcover v200 (2021) class legend
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Subset relevant for Switzerland
ESA_CLASSES_SWITZERLAND = {
    10: "Tree cover",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    70: "Snow and ice",
    80: "Permanent water bodies",
}


def get_worldcover_tiles(bounds=None, version="v200", year="2021"):
    """
    Return S3 paths for ESA Worldcover tiles covering *bounds*.

    Tiles are on a 3° grid, named like
    ``ESA_WorldCover_10m_2021_v200_N45E006_Map.tif``.
    The tile label uses the SW corner snapped to the nearest multiple of 3.
    """
    b = bounds or BOUNDS

    def _snap(val, step):
        """Snap down to nearest multiple of *step*."""
        return int(np.floor(val / step) * step)

    lat_lo = _snap(b["lat_min"], 3)
    lat_hi = _snap(b["lat_max"], 3)
    lon_lo = _snap(b["lon_min"], 3)
    lon_hi = _snap(b["lon_max"], 3)

    tiles = []
    for lat in range(lat_lo, lat_hi + 1, 3):
        for lon in range(lon_lo, lon_hi + 1, 3):
            ns = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            ew = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            name = f"ESA_WorldCover_10m_{year}_{version}_{ns}{ew}_Map.tif"
            tiles.append(f"s3://esa-worldcover/{version}/{year}/map/{name}")
    return tiles


def compute_landcover_fractions(
    target_grid,
    class_dict=None,
    bounds=None,
):
    """
    Compute per-class land-cover fractions on the 30 m master grid.

    Uses rasterio windowed reads and ``rasterio.warp.reproject`` directly
    on numpy arrays to keep memory usage low.  Each tile is read once,
    and for each class a uint8 binary mask is reprojected with
    ``Resampling.average`` to produce a fraction in [0, 1] at 30 m.

    Parameters
    ----------
    target_grid : xarray.DataArray
        The 30 m master grid (e.g. the DEM).
    class_dict : dict[int, str] | None
        ``{class_id: class_name}``.  Defaults to Swiss-relevant classes.
    bounds : dict | None
        Spatial bounds.  Defaults to ``BOUNDS``.

    Returns
    -------
    xarray.DataArray  (band=n_classes, y, x) with values in [0, 1].
    """
    from rasterio.warp import reproject, calculate_default_transform
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    b = bounds or BOUNDS
    class_dict = class_dict or ESA_CLASSES_SWITZERLAND
    tile_paths = get_worldcover_tiles(b)
    class_ids = list(class_dict.keys())
    class_names = list(class_dict.values())

    # Target grid properties (30 m)
    tgt_band = target_grid.sel(band=1) if "band" in target_grid.dims else target_grid
    dst_height, dst_width = tgt_band.shape
    dst_crs = target_grid.rio.crs
    dst_transform = target_grid.rio.transform()

    # Accumulators at 30 m — tiny compared to 10 m source
    accum = np.zeros((len(class_ids), dst_height, dst_width), dtype=np.float32)
    count = np.zeros((dst_height, dst_width), dtype=np.float32)

    print(f"Processing {len(tile_paths)} tiles × {len(class_ids)} classes "
          f"→ {dst_height}×{dst_width} target grid")

    env = rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        AWS_DEFAULT_REGION="eu-central-1",
    )

    with env:
        for ti, tile_path in enumerate(tile_paths):
            print(f"  Tile {ti + 1}/{len(tile_paths)}: "
                  f"{os.path.basename(tile_path)}")

            with rasterio.open(tile_path) as src:
                # Compute the window that covers our bounds within this tile
                try:
                    win = window_from_bounds(
                        b["lon_min"], b["lat_min"],
                        b["lon_max"], b["lat_max"],
                        transform=src.transform,
                    )
                    # Clamp to valid tile extent
                    win = win.intersection(
                        rasterio.windows.Window(0, 0, src.width, src.height)
                    )
                except Exception:
                    print("    (no overlap, skipping)")
                    continue

                if win.width < 1 or win.height < 1:
                    print("    (no overlap, skipping)")
                    continue

                # Read only the windowed region — uint8, ~manageable
                src_data = src.read(1, window=win)
                src_transform = src.window_transform(win)
                src_crs = src.crs

            print(f"    Read window: {src_data.shape[0]}×{src_data.shape[1]} px")

            # For each class, build a binary mask and reproject
            for ci, class_id in enumerate(class_ids):
                mask = (src_data == class_id).astype(np.float32)

                dst_buf = np.zeros((dst_height, dst_width), dtype=np.float32)
                reproject(
                    source=mask,
                    destination=dst_buf,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.average,
                )
                accum[ci] += dst_buf
                del mask, dst_buf

            # Coverage: reproject an all-ones array to know where this tile
            # contributed data
            ones = np.ones_like(src_data, dtype=np.float32)
            cov_buf = np.zeros((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=ones,
                destination=cov_buf,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
            )
            count += (cov_buf > 0).astype(np.float32)

            del src_data, ones, cov_buf
            gc.collect()
            print(f"    Done.")

    # Average overlapping tiles
    count_safe = np.clip(count, 1, None)
    for ci in range(len(class_ids)):
        accum[ci] /= count_safe

    # Wrap back into xarray with proper coordinates
    result = xr.DataArray(
        accum,
        dims=["band", tgt_band.dims[0], tgt_band.dims[1]],
        coords={
            "band": class_names,
            tgt_band.dims[0]: tgt_band.coords[tgt_band.dims[0]],
            tgt_band.dims[1]: tgt_band.coords[tgt_band.dims[1]],
        },
    )
    result = result.rio.write_crs(dst_crs)
    result = result.rio.write_transform(dst_transform)

    print(f"  Land-cover fractions: {dict(result.sizes)}")
    return result


# ---------------------------------------------------------------------------
# ETH Global Canopy Height (10 m, 2020)
# ---------------------------------------------------------------------------

ETH_CANOPY_BASE = (
    "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download"
    "?path=/3deg_cogs&files="
)


def get_canopy_height_tiles(bounds=None):
    """
    Return download URLs for ETH Global Canopy Height 10 m tiles
    covering *bounds*.  Same 3°×3° grid as ESA Worldcover.
    """
    b = bounds or BOUNDS

    def _snap(val, step):
        return int(np.floor(val / step) * step)

    lat_lo = _snap(b["lat_min"], 3)
    lat_hi = _snap(b["lat_max"], 3)
    lon_lo = _snap(b["lon_min"], 3)
    lon_hi = _snap(b["lon_max"], 3)

    tiles = []
    for lat in range(lat_lo, lat_hi + 1, 3):
        for lon in range(lon_lo, lon_hi + 1, 3):
            ns = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            ew = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            name = f"ETH_GlobalCanopyHeight_10m_2020_{ns}{ew}_Map.tif"
            tiles.append(ETH_CANOPY_BASE + name)
    return tiles


def compute_canopy_height(target_grid, bounds=None):
    """
    Load ETH canopy height tiles, regrid to the 30 m master grid using
    bilinear resampling, and return a single DataArray.

    Uses the same memory-efficient tile-by-tile rasterio approach as
    ``compute_landcover_fractions``.

    Parameters
    ----------
    target_grid : xarray.DataArray  (the 30 m DEM)
    bounds : dict | None

    Returns
    -------
    xarray.DataArray (y, x) — canopy height in metres on the 30 m grid.
    """
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds as window_from_bounds

    b = bounds or BOUNDS
    tile_urls = get_canopy_height_tiles(b)

    tgt_band = target_grid.sel(band=1) if "band" in target_grid.dims else target_grid
    dst_height, dst_width = tgt_band.shape
    dst_crs = target_grid.rio.crs
    dst_transform = target_grid.rio.transform()

    accum = np.zeros((dst_height, dst_width), dtype=np.float64)
    count = np.zeros((dst_height, dst_width), dtype=np.float64)

    print(f"Processing {len(tile_urls)} canopy height tiles → "
          f"{dst_height}×{dst_width} target grid")

    for ti, url in enumerate(tile_urls):
        fname = url.split("files=")[-1]
        print(f"  Tile {ti + 1}/{len(tile_urls)}: {fname}")

        vsi_path = f"/vsicurl/{url}"
        try:
            with rasterio.open(vsi_path) as src:
                try:
                    win = window_from_bounds(
                        b["lon_min"], b["lat_min"],
                        b["lon_max"], b["lat_max"],
                        transform=src.transform,
                    )
                    win = win.intersection(
                        rasterio.windows.Window(0, 0, src.width, src.height)
                    )
                except Exception:
                    print("    (no overlap, skipping)")
                    continue

                if win.width < 1 or win.height < 1:
                    print("    (no overlap, skipping)")
                    continue

                src_data = src.read(1, window=win).astype(np.float32)
                src_transform = src.window_transform(win)
                src_crs = src.crs

                # Mask nodata (255 = built-up/snow/ice/water in ETH dataset)
                src_data[src_data == 255] = 0.0

            print(f"    Read window: {src_data.shape[0]}×{src_data.shape[1]} px")

            dst_buf = np.zeros((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=src_data,
                destination=dst_buf,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

            valid = dst_buf > 0
            accum += dst_buf
            count += valid.astype(np.float64)

            del src_data, dst_buf
            gc.collect()
            print("    Done.")

        except Exception as e:
            print(f"    Error: {e}")

    count_safe = np.clip(count, 1, None)
    result_arr = (accum / count_safe).astype(np.float32)

    result = xr.DataArray(
        result_arr,
        dims=[tgt_band.dims[0], tgt_band.dims[1]],
        coords={
            tgt_band.dims[0]: tgt_band.coords[tgt_band.dims[0]],
            tgt_band.dims[1]: tgt_band.coords[tgt_band.dims[1]],
        },
        name="canopy_height",
    )
    result = result.rio.write_crs(dst_crs)
    result = result.rio.write_transform(dst_transform)

    print(f"  Canopy height: {dict(result.sizes)}")
    return result


# ---------------------------------------------------------------------------
# GIMMS MODIS NDVI (250 m, 8-day composites → annual max)
# ---------------------------------------------------------------------------

GIMMS_NDVI_BASE = "https://gimms.gsfc.nasa.gov/MODIS/std/GMOD09Q1/cog/NDVI"


def compute_ndvi_annual_max(
    target_grid,
    bounds=None,
    year=2020,
    doy_range=(97, 273),
):
    """
    Compute annual maximum NDVI from GIMMS MODIS 8-day composites,
    regridded to the 30 m master grid.

    Reads each global COG via ``/vsicurl/`` with a windowed read for
    Switzerland only, so memory stays low.  Takes the per-pixel max
    across the growing season, then regrids with bilinear resampling.

    Parameters
    ----------
    target_grid : xarray.DataArray  (the 30 m DEM)
    bounds : dict | None
    year : int
    doy_range : tuple (start_doy, end_doy) inclusive

    Returns
    -------
    xarray.DataArray (y, x) — NDVI [0–1] on the 30 m grid.
    """
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds as window_from_bounds

    b = bounds or BOUNDS

    # Build list of DOYs in the growing season
    all_doys = list(range(1, 366, 8))
    doys = [d for d in all_doys if doy_range[0] <= d <= doy_range[1]]

    tgt_band = target_grid.sel(band=1) if "band" in target_grid.dims else target_grid
    dst_height, dst_width = tgt_band.shape
    dst_crs = target_grid.rio.crs
    dst_transform = target_grid.rio.transform()

    # Accumulate max NDVI at 250 m (native), then regrid once at the end
    ndvi_max = None

    print(f"Processing {len(doys)} GIMMS NDVI composites for {year} "
          f"(DOY {doy_range[0]}–{doy_range[1]})")

    for di, doy in enumerate(doys):
        fname = f"GMOD09Q1.A{year}{doy:03d}.08d.latlon.global.061.NDVI.tif"
        url = f"{GIMMS_NDVI_BASE}/{year}/{doy:03d}/{fname}"
        vsi = f"/vsicurl/{url}"

        print(f"  [{di+1}/{len(doys)}] DOY {doy}...", end=" ", flush=True)

        try:
            with rasterio.open(vsi) as src:
                win = window_from_bounds(
                    b["lon_min"], b["lat_min"],
                    b["lon_max"], b["lat_max"],
                    transform=src.transform,
                )
                win = win.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height)
                )
                data = src.read(1, window=win)
                src_transform = src.window_transform(win)
                src_crs = src.crs

            # Convert uint8 [0-250] → float [0-1], mask invalid (>250)
            valid = data <= 250
            ndvi = np.where(valid, data * 0.004, 0.0).astype(np.float32)

            if ndvi_max is None:
                ndvi_max = ndvi.copy()
                ndvi_src_transform = src_transform
                ndvi_src_crs = src_crs
            else:
                ndvi_max = np.maximum(ndvi_max, ndvi)

            del data, ndvi
            print("ok")

        except Exception as e:
            print(f"error: {e}")

    if ndvi_max is None:
        raise RuntimeError("No NDVI composites could be read")

    print(f"  Max composite shape: {ndvi_max.shape}, "
          f"range: {ndvi_max.min():.3f}–{ndvi_max.max():.3f}")

    # Regrid the 250 m max composite to 30 m
    print("  Regridding to 30 m master grid...")
    dst_buf = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=ndvi_max,
        destination=dst_buf,
        src_transform=ndvi_src_transform,
        src_crs=ndvi_src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    result = xr.DataArray(
        dst_buf,
        dims=[tgt_band.dims[0], tgt_band.dims[1]],
        coords={
            tgt_band.dims[0]: tgt_band.coords[tgt_band.dims[0]],
            tgt_band.dims[1]: tgt_band.coords[tgt_band.dims[1]],
        },
        name="ndvi_max",
    )
    result = result.rio.write_crs(dst_crs)
    result = result.rio.write_transform(dst_transform)

    del ndvi_max
    gc.collect()

    print(f"  NDVI annual max: {dict(result.sizes)}")
    return result


# ---------------------------------------------------------------------------
# Human Footprint Index (Mu et al. 2022, 1 km, Mollweide)
# ---------------------------------------------------------------------------

HFP_FIGSHARE_URL = "https://ndownloader.figshare.com/files/30716126"  # hfp2009.zip


def compute_human_footprint(target_grid, bounds=None, url=None):
    """
    Download the global Human Footprint Index (Mu et al. 2022),
    reproject from Mollweide to WGS84, and regrid to the 30 m master grid.

    The dataset is ~457 MB zipped.  It is downloaded to a temp directory,
    unzipped, reprojected, and cleaned up.

    Parameters
    ----------
    target_grid : xarray.DataArray  (the 30 m DEM)
    bounds : dict | None
    url : str | None  override the default Figshare URL

    Returns
    -------
    xarray.DataArray (y, x) — HFP score [0–50] on the 30 m grid.
    """
    import tempfile
    import zipfile
    import urllib.request
    from rasterio.warp import reproject, calculate_default_transform
    from rasterio.windows import from_bounds as window_from_bounds

    b = bounds or BOUNDS
    url = url or HFP_FIGSHARE_URL

    tgt_band = target_grid.sel(band=1) if "band" in target_grid.dims else target_grid
    dst_height, dst_width = tgt_band.shape
    dst_crs = target_grid.rio.crs
    dst_transform = target_grid.rio.transform()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "hfp.zip")

        print(f"  Downloading HFP ({url.split('/')[-1]})...")
        urllib.request.urlretrieve(url, zip_path)

        print("  Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"No .tif found in zip: {zf.namelist()}")
            zf.extract(tif_names[0], tmpdir)
            tif_path = os.path.join(tmpdir, tif_names[0])

        print(f"  Reading {tif_names[0]}...")
        with rasterio.open(tif_path) as src:
            # The HFP is in Mollweide — we need to reproject the bounds
            # to Mollweide to do a windowed read, or just read the full
            # raster (it's 1 km so the full global grid is manageable).
            data = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_transform = src.transform

            # Mask nodata
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan

        print(f"  Source shape: {data.shape}, CRS: {src_crs}")
        print("  Reprojecting to 30 m master grid...")

        dst_buf = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=data,
            destination=dst_buf,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        del data
        gc.collect()

    result = xr.DataArray(
        dst_buf,
        dims=[tgt_band.dims[0], tgt_band.dims[1]],
        coords={
            tgt_band.dims[0]: tgt_band.coords[tgt_band.dims[0]],
            tgt_band.dims[1]: tgt_band.coords[tgt_band.dims[1]],
        },
        name="human_footprint",
    )
    result = result.rio.write_crs(dst_crs)
    result = result.rio.write_transform(dst_transform)

    print(f"  Human Footprint: {dict(result.sizes)}, "
          f"range: {float(np.nanmin(dst_buf)):.1f}–{float(np.nanmax(dst_buf)):.1f}")
    return result
