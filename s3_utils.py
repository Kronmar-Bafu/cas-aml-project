"""
S3 helper utilities for the GNN-SDM pipeline.
"""

import os

import boto3
import requests


def stream_to_s3(file_list_path: str, bucket_name: str, s3_prefix: str = "raw/chelsa/"):
    """
    Stream-download files listed in *file_list_path* directly into an S3 bucket,
    skipping files that already exist.
    """
    s3_client = boto3.client("s3")

    with open(file_list_path) as f:
        links = [line.strip() for line in f if line.strip()]

    print(f"Streaming {len(links)} files → s3://{bucket_name}/{s3_prefix}")

    for i, url in enumerate(links, 1):
        filename = url.split("/")[-1]
        s3_key = os.path.join(s3_prefix, filename)

        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            print(f"[{i}/{len(links)}] Already exists: {filename}")
            continue
        except Exception:
            pass

        print(f"[{i}/{len(links)}] Streaming: {filename}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                s3_client.upload_fileobj(r.raw, bucket_name, s3_key)
            print(f"  Done: {filename}")
        except Exception as e:
            print(f"  ERROR {filename}: {e}")

    print("Streaming complete.")


# ---------------------------------------------------------------------------
# Zarr save / load helpers
# ---------------------------------------------------------------------------


def save_zarr(data, s3_zarr_path, name="data", chunks=None, s3=None):
    """
    Save an xarray DataArray or Dataset to S3 as a chunked Zarr store.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
    s3_zarr_path : str  e.g. "s3://bucket/path/to/store.zarr"
    name : str  variable name (only used if *data* is a DataArray)
    chunks : dict | None  chunk sizes, defaults to {"x": 512, "y": 512}
    s3 : s3fs.S3FileSystem | None
    """
    import xarray as xr
    import s3fs as _s3fs

    s3 = s3 or _s3fs.S3FileSystem(anon=False)
    store = _s3fs.S3Map(root=s3_zarr_path, s3=s3, check=False)
    chunks = chunks or {"x": 512, "y": 512}

    ds = data if isinstance(data, xr.Dataset) else data.to_dataset(name=name)
    ds = ds.chunk(chunks)
    ds.to_zarr(store=store, mode="w", consolidated=True)
    print(f"Saved → {s3_zarr_path}")


def load_zarr(s3_zarr_path, name=None, s3=None):
    """
    Load an xarray Dataset (or a single variable) from a Zarr store on S3.

    Parameters
    ----------
    s3_zarr_path : str
    name : str | None  if given, return just that DataArray instead of the Dataset
    s3 : s3fs.S3FileSystem | None

    Returns
    -------
    xarray.Dataset or xarray.DataArray
    """
    import xarray as xr
    import s3fs as _s3fs

    s3 = s3 or _s3fs.S3FileSystem(anon=False)
    store = _s3fs.S3Map(root=s3_zarr_path, s3=s3, check=False)
    ds = xr.open_zarr(store, consolidated=True)
    if name:
        return ds[name]
    return ds
