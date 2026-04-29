"""ATLAS database access and cache helpers.

The ATLAS protein MD corpus is large enough that training should use a local
cache or an object-store mirror staged to local disk.  This module keeps the
download layer small and deterministic: IDs are normalized, URLs are explicit,
and every download can be recorded in a JSON manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ATLAS_API_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
ATLAS_STATIC_BASE = "https://www.dsimb.inserm.fr/ATLAS/database"

_ATLAS_ID_RE = re.compile(r"^(?P<pdb>[0-9A-Za-z]{4})[_:-](?P<chain>[A-Za-z0-9])$")


@dataclass(frozen=True)
class AtlasDownloadRecord:
    """One ATLAS download/cache manifest entry."""

    atlas_id: str
    url: str
    path: Path
    size_bytes: int
    sha256: str
    status: str
    error: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def normalize_atlas_id(value: str) -> str:
    """Normalize a PDB-chain identifier to ATLAS ``pdbid_Chain`` form.

    Examples
    --------
    ``"16PK:a"`` and ``"16pk_A"`` both normalize to ``"16pk_A"``.
    """
    text = value.strip()
    # Allow inline comments in plain text ID lists.
    text = text.split("#", 1)[0].strip()
    match = _ATLAS_ID_RE.match(text)
    if match is None:
        raise ValueError(
            f"Malformed ATLAS ID {value!r}; expected four-character PDB ID plus "
            "single chain, e.g. '16pk_A'."
        )
    pdb_id = match.group("pdb").lower()
    chain_id = match.group("chain").upper()
    return f"{pdb_id}_{chain_id}"


def atlas_api_url(atlas_id: str, *, dataset: str = "ATLAS", bundle: str = "protein") -> str:
    """Return the ATLAS REST API URL for a bundle."""
    normalized = normalize_atlas_id(atlas_id)
    dataset_part = _normalize_dataset(dataset)
    bundle_part = _normalize_bundle(bundle)
    return f"{ATLAS_API_BASE}/{dataset_part}/{bundle_part}/{normalized}"


def atlas_static_url(atlas_id: str, *, dataset: str = "ATLAS") -> str:
    """Return the static protein ZIP URL used by AlphaFlow/MDGen-style scripts."""
    normalized = normalize_atlas_id(atlas_id)
    dataset_part = _normalize_dataset(dataset)
    return f"{ATLAS_STATIC_BASE}/{dataset_part}/{normalized}/{normalized}_protein.zip"


def cache_zip_path(raw_dir: Path, atlas_id: str) -> Path:
    """Return the deterministic local path for a protein ZIP."""
    normalized = normalize_atlas_id(atlas_id)
    return Path(raw_dir).expanduser() / normalized / f"{normalized}_protein.zip"


def read_ids_file(path: Path, *, limit: int | None = None) -> list[str]:
    """Read ATLAS IDs from a newline-delimited text file.

    Blank lines and comments beginning with ``#`` are ignored.  Inline comments
    are also supported.
    """
    ids: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        ids.append(normalize_atlas_id(line))
        if limit is not None and len(ids) >= limit:
            break
    return ids


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a local file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, records: Iterable[AtlasDownloadRecord]) -> None:
    """Write a JSON manifest for downloaded/cache-hit ATLAS bundles."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_unix": time.time(),
        "records": [record.to_json_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_atlas_zip(
    atlas_id: str,
    *,
    raw_dir: Path,
    dataset: str = "ATLAS",
    url_source: str = "static",
    overwrite: bool = False,
    timeout: float = 60.0,
) -> AtlasDownloadRecord:
    """Download one ATLAS protein ZIP into the local cache.

    Parameters
    ----------
    atlas_id:
        ATLAS identifier such as ``16pk_A``.
    raw_dir:
        Root directory for cached ZIPs.
    dataset:
        ATLAS dataset namespace.  ``ATLAS`` is the default.
    url_source:
        ``"static"`` for the stable static ZIP pattern or ``"api"`` for the
        REST endpoint.
    overwrite:
        Re-download even when the target file exists.
    timeout:
        Per-request timeout in seconds for the Python ``requests`` backend.
    """
    normalized = normalize_atlas_id(atlas_id)
    target = cache_zip_path(raw_dir, normalized)
    url = (
        atlas_api_url(normalized, dataset=dataset, bundle="protein")
        if url_source == "api"
        else atlas_static_url(normalized, dataset=dataset)
    )
    if url_source not in {"api", "static"}:
        raise ValueError("url_source must be 'api' or 'static'")

    if target.is_file() and not overwrite:
        return AtlasDownloadRecord(
            atlas_id=normalized,
            url=url,
            path=target,
            size_bytes=target.stat().st_size,
            sha256=sha256_file(target),
            status="cached",
        )

    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Downloading ATLAS bundles requires the 'requests' package.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(target)
        return AtlasDownloadRecord(
            atlas_id=normalized,
            url=url,
            path=target,
            size_bytes=target.stat().st_size,
            sha256=sha256_file(target),
            status="downloaded",
        )
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return AtlasDownloadRecord(
            atlas_id=normalized,
            url=url,
            path=target,
            size_bytes=0,
            sha256="",
            status="failed",
            error=str(exc),
        )


def _normalize_dataset(dataset: str) -> str:
    mapping = {
        "atlas": "ATLAS",
        "chameleon": "chameleon",
        "dpf": "DPF",
    }
    key = dataset.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported ATLAS dataset {dataset!r}; expected ATLAS, chameleon, or DPF.")
    return mapping[key]


def _normalize_bundle(bundle: str) -> str:
    key = bundle.strip().lower()
    if key not in {"protein", "analysis", "total"}:
        raise ValueError(f"Unsupported ATLAS bundle {bundle!r}; expected protein, analysis, or total.")
    return key
