"""
Plotting utilities for the GNN-SDM pipeline.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import s3fs


def plot_raster(data, title="", *, cmap="terrain", step=20, vmin=None, vmax=None,
                label="", figsize=(12, 7), robust=False):
    """
    Quick plot of any 2-D xarray DataArray on a cartopy map.

    Downsamples by *step* (every Nth pixel) so the kernel doesn't OOM
    on full-resolution 30 m grids.  Works for DEM, slope, single LC
    fraction bands, CHELSA layers, etc.

    Parameters
    ----------
    data : xarray.DataArray  (y, x) — no band dim expected
    title : str
    cmap : str
    step : int   downsample factor (default 20)
    vmin, vmax : float | None
    label : str  colorbar label
    figsize : tuple
    robust : bool  if True, use 2nd/98th percentile for color limits
    """
    from geo_utils import apply_map_decor

    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = data[::step, ::step].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
    )

    apply_map_decor(ax, title)
    plt.colorbar(im, label=label, fraction=0.02, pad=0.04)
    plt.show()


def plot_roi(layers, roi, *, suptitle="", cols=4, figsize_per_panel=(5, 4)):
    """
    Plot a grid of raster layers clipped to a region of interest.

    Useful for quick visual QC of all predictor variables in a small area.

    Parameters
    ----------
    layers : dict[str, xarray.DataArray]
        ``{title: DataArray}`` — each DataArray should have ``y`` and ``x``
        dimensions (standard rioxarray / xarray naming).
    roi : dict
        ``{"lat_min", "lat_max", "lon_min", "lon_max"}``
    suptitle : str
        Figure super-title.
    cols : int
        Number of columns in the grid.
    figsize_per_panel : tuple
        ``(width, height)`` per subplot.
    """
    import math

    n = len(layers)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = list(axes.flat) if n > 1 else [axes]

    for i, (title, da) in enumerate(layers.items()):
        ax = axes[i]
        subset = da.sel(
            y=slice(roi["lat_max"], roi["lat_min"]),
            x=slice(roi["lon_min"], roi["lon_max"]),
        )
        subset.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=True, robust=True)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5)
        ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.3, color="blue")
        ax.set_extent([roi["lon_min"], roi["lon_max"], roi["lat_min"], roi["lat_max"]])
        ax.set_title(title, fontsize=9)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_bio_from_zarr(s3_path, var_name, title, *, s3=None, cmap="YlGnBu", unit="mm"):
    """
    Plot a bioclimatic variable directly from a Zarr store on S3.

    Loads the variable lazily and plots it on a cartopy map with borders,
    lakes, and rivers. Uses robust color scaling to handle outliers.

    Parameters
    ----------
    s3_path : str
        S3 path to the Zarr store (e.g. "s3://bucket/chelsa/bio1_30m.zarr").
    var_name : str
        Variable name inside the Zarr Dataset.
    title : str
        Plot title.
    s3 : s3fs.S3FileSystem | None
        Optional pre-configured filesystem.
    cmap : str
        Matplotlib colormap name.
    unit : str
        Unit label for the colorbar.
    """
    s3 = s3 or s3fs.S3FileSystem(anon=False)
    store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
    ds = xr.open_zarr(store, consolidated=True)

    plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ds[var_name].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        robust=True,
        add_colorbar=False,
    )

    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.5, edgecolor="black")
    ax.add_feature(cfeature.LAKES, alpha=0.4, facecolor="cyan")
    ax.add_feature(cfeature.RIVERS, alpha=0.3, edgecolor="blue")
    ax.set_extent([5.9, 10.5, 45.8, 47.9], crs=ccrs.PlateCarree())

    plt.title(title, fontsize=15, pad=20, fontweight="bold")
    plt.colorbar(im, label=f"{var_name} [{unit}]", fraction=0.02, pad=0.04)
    plt.show()


def plot_landcover_fractions(lc_data, cols=3, step=20):
    """
    Plot land-cover fraction maps for every class in *lc_data*.

    Creates a grid of maps, one per land-cover class on the ``band``
    dimension. Each map shows the fraction [0, 1] of that class within
    each 30 m cell. Downsampled by *step* to avoid OOM.

    Parameters
    ----------
    lc_data : xarray.DataArray
        Must have a ``band`` dimension with class names as coordinates,
        and ``y``, ``x`` spatial dimensions. Values in [0, 1].
    cols : int
        Number of columns in the subplot grid.
    step : int
        Downsample factor (every Nth pixel rendered).
    """
    import math

    bands = list(lc_data.band.values)
    n = len(bands)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(6 * cols, 5 * rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = list(axes.flat) if n > 1 else [axes]

    for i, band_name in enumerate(bands):
        ax = axes[i]
        data = lc_data.sel(band=band_name)[::step, ::step]

        im = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="YlGnBu",
            add_colorbar=False,
            vmin=0,
            vmax=1,
        )

        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
        ax.set_title(str(band_name), fontsize=11)
        plt.colorbar(im, ax=ax, label="Fraction", fraction=0.046, pad=0.04)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
