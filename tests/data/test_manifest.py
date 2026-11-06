import hashlib

from kairos.data import manifest, store


def test_manifest_roundtrip(root, make_dataset):
    store.write_dataset(make_dataset("c.a"), root)
    m = manifest.write_manifest(root)
    rel = "processed/series/c.a.parquet"
    assert set(m["files"]) == {rel}
    raw = (root / rel).read_bytes()
    assert m["files"][rel]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest.verify_manifest(root) == []


def test_digest_stable_and_sensitive(root, make_dataset):
    store.write_dataset(make_dataset("c.a"), root)
    first = manifest.write_manifest(root)
    again = manifest.write_manifest(root)
    assert again == first  # unchanged dataset keeps the file as is
    store.write_dataset(make_dataset("c.a", growth=0.1), root)
    store.write_dataset(make_dataset("c.b"), root)
    assert sorted(manifest.verify_manifest(root)) == [
        "changed: processed/series/c.a.parquet",
        "untracked: processed/series/c.b.parquet",
    ]
    assert manifest.write_manifest(root)["digest"] != first["digest"]


def test_missing_reported(root, make_dataset):
    store.write_dataset(make_dataset("c.a"), root)
    manifest.write_manifest(root)
    (root / "processed/series/c.a.parquet").unlink()
    assert manifest.verify_manifest(root) == ["missing: processed/series/c.a.parquet"]


def test_empty(root):
    m = manifest.build_manifest(root)
    assert m["files"] == {}
    assert m["digest"] == hashlib.sha256(b"").hexdigest()
