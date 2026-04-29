"""Tests for ATLAS download/cache helpers."""

from __future__ import annotations

import json

import pytest

from genai_tps.data.atlas import (
    AtlasDownloadRecord,
    atlas_api_url,
    atlas_static_url,
    cache_zip_path,
    normalize_atlas_id,
    read_ids_file,
    write_manifest,
)


def test_normalize_atlas_id_accepts_common_forms():
    assert normalize_atlas_id("16pk_A") == "16pk_A"
    assert normalize_atlas_id("16PK:A") == "16pk_A"
    assert normalize_atlas_id("  1abc-a  ") == "1abc_A"


@pytest.mark.parametrize("bad_id", ["", "abc_A", "12345_A", "1abc_", "1abc_AB"])
def test_normalize_atlas_id_rejects_malformed_values(bad_id: str):
    with pytest.raises(ValueError):
        normalize_atlas_id(bad_id)


def test_url_construction_uses_normalized_ids():
    assert (
        atlas_api_url("16PK:a")
        == "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/16pk_A"
    )
    assert (
        atlas_static_url("16PK:a")
        == "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/16pk_A/16pk_A_protein.zip"
    )


def test_cache_zip_path_is_deterministic(tmp_path):
    assert cache_zip_path(tmp_path, "16PK:a") == tmp_path / "16pk_A" / "16pk_A_protein.zip"


def test_read_ids_file_ignores_comments_and_limits(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\n16pk_A\n\n1abc:B  # inline comment\n2def_C\n", encoding="utf-8")

    assert read_ids_file(ids_file, limit=2) == ["16pk_A", "1abc_B"]


def test_write_manifest_round_trips_records(tmp_path):
    manifest = tmp_path / "manifest.json"
    records = [
        AtlasDownloadRecord(
            atlas_id="16pk_A",
            url="https://example.test/16pk_A.zip",
            path=tmp_path / "16pk_A.zip",
            size_bytes=123,
            sha256="abc",
            status="downloaded",
        )
    ]

    write_manifest(manifest, records)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["records"][0]["atlas_id"] == "16pk_A"
    assert payload["records"][0]["size_bytes"] == 123
    assert payload["records"][0]["status"] == "downloaded"
