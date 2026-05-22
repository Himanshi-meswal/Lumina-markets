"""I/O helpers: load the four case-study tables and persist node artifacts.

Keeping all disk access here means nodes receive and return DataFrames only,
which makes them trivial to unit-test with synthetic frames.

Cloud-aware: any path beginning with 'gs://' is read from / written to Google
Cloud Storage. This lets the SAME code run locally (Excel on disk) or on GCP
(Excel + artifacts in a bucket) with no node changes — only config paths differ.
Reading gs:// Excel needs `gcsfs`; artifact gs:// I/O needs `google-cloud-storage`.
"""
from __future__ import annotations
import os
import io
import pickle
import pandas as pd

from . import config


def _is_gcs(path: str) -> bool:
    return isinstance(path, str) and path.startswith("gs://")


def _split_gcs(uri: str):
    """gs://bucket/some/key -> ('bucket', 'some/key')."""
    rest = uri[len("gs://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def load_tables(excel_path: str | None = None) -> dict[str, pd.DataFrame]:
    """Read all sheets, normalise dates, drop dimension rows with no sales.

    `excel_path` may be a local path OR a gs:// URI (pandas reads gs:// directly
    when gcsfs is installed). Returns a dict keyed by 'sales', 'product',
    'store', 'pricing', 'calendar'.
    """
    path = excel_path or config.EXCEL_PATH
    tables = {
        key: pd.read_excel(path, sheet_name=sheet)
        for key, sheet in config.SHEETS.items()
    }
    for key in ("sales", "pricing", "calendar"):
        if key in tables:
            tables[key]["Date"] = pd.to_datetime(tables[key]["Date"])

    # internal consistency: keep only stores that appear in sales
    used = set(tables["sales"]["Store_ID"].unique())
    tables["store"] = tables["store"][tables["store"].Store_ID.isin(used)].reset_index(drop=True)

    tables["sales"] = tables["sales"].sort_values(
        ["SKU_ID", "Store_ID", "Date"]
    ).reset_index(drop=True)
    return tables


def _gcs_write_bytes(uri: str, data: bytes):
    from google.cloud import storage
    bucket_name, key = _split_gcs(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(key).upload_from_string(data)


def _gcs_read_bytes(uri: str) -> bytes:
    from google.cloud import storage
    bucket_name, key = _split_gcs(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(key).download_as_bytes()


def _gcs_exists(uri: str) -> bool:
    from google.cloud import storage
    bucket_name, key = _split_gcs(uri)
    client = storage.Client()
    return client.bucket(bucket_name).blob(key).exists()


def save_artifact(obj, name: str, artifact_dir: str | None = None) -> str:
    """Pickle an intermediate result. Writes to GCS if artifact_dir is gs://."""
    d = artifact_dir or config.ARTIFACT_DIR
    if _is_gcs(d):
        uri = f"{d.rstrip('/')}/{name}.pkl"
        _gcs_write_bytes(uri, pickle.dumps(obj))
        return uri
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f"{name}.pkl")
    with open(fp, "wb") as f:
        pickle.dump(obj, f)
    return fp


def load_artifact(name: str, artifact_dir: str | None = None):
    d = artifact_dir or config.ARTIFACT_DIR
    if _is_gcs(d):
        uri = f"{d.rstrip('/')}/{name}.pkl"
        return pickle.loads(_gcs_read_bytes(uri))
    fp = os.path.join(d, f"{name}.pkl")
    with open(fp, "rb") as f:
        return pickle.load(f)


def artifact_exists(name: str, artifact_dir: str | None = None) -> bool:
    d = artifact_dir or config.ARTIFACT_DIR
    if _is_gcs(d):
        return _gcs_exists(f"{d.rstrip('/')}/{name}.pkl")
    return os.path.exists(os.path.join(d, f"{name}.pkl"))
