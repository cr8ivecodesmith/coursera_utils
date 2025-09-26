from __future__ import annotations

import json
import sys
import types
import zipfile

import pytest

from study_utils.rag import vector_store


def _dummy_manifest(name: str) -> vector_store.VectorStoreManifest:
    documents = (
        vector_store.SourceDocument(
            source_path="doc1.md",
            checksum="abc",
            size_bytes=10,
            chunk_count=2,
        ),
    )
    return vector_store.build_manifest(
        name=name,
        embedding=vector_store.EmbeddingMetadata(
            provider="openai",
            model="test-model",
            dimension=3,
        ),
        chunking=vector_store.ChunkingMetadata(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=10,
            token_overlap=2,
            fallback_delimiter="\n\n",
        ),
        dedupe=vector_store.DedupMetadata(
            strategy="checksum",
            checksum_algorithm="sha256",
        ),
        documents=documents,
    )


def test_build_manifest_totals():
    manifest = _dummy_manifest("physics")
    assert manifest.total_chunks == 2
    payload = manifest.to_dict()
    assert payload["embedding"]["model"] == "test-model"
    rebuilt = vector_store.VectorStoreManifest.from_dict(payload)
    assert rebuilt.name == manifest.name
    assert rebuilt.chunking.fallback_delimiter == "\n\n"


def test_repository_prepare_write_list(tmp_path):
    root = tmp_path / "stores"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    repo.prepare_store("physics")
    manifest = _dummy_manifest("physics")
    repo.write_manifest(manifest)
    manifests = repo.list_manifests()
    assert len(manifests) == 1
    assert manifests[0].name == "physics"


def test_repository_prepare_requires_valid_name(tmp_path):
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    with pytest.raises(vector_store.VectorStoreError):
        repo.prepare_store("Invalid Name!")


def test_export_import_round_trip(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    backend = vector_store.InMemoryVectorStoreBackend()
    store_dir = repo.prepare_store("physics")
    backend.create(
        store_dir,
        texts=["chunk"],
        embeddings=[[0.0, 1.0]],
        metadatas=[{"source_path": "doc", "chunk_index": 0}],
    )
    manifest = _dummy_manifest("physics")
    repo.write_manifest(manifest)

    archive = tmp_path / "physics.zip"
    repo.export_store("physics", archive)
    repo.delete("physics")

    imported = repo.import_store("physics", archive)
    assert imported.name == "physics"
    assert imported.total_chunks == manifest.total_chunks


def test_import_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "oops")
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    repo.ensure_root()
    with pytest.raises(vector_store.VectorStoreError):
        repo.import_store("physics", archive)


def test_load_manifest_rejects_corruption(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    store_dir = repo.prepare_store("physics")
    bad_manifest = store_dir / "manifest.json"
    bad_manifest.write_text("not json", encoding="utf-8")
    with pytest.raises(vector_store.VectorStoreError):
        repo.load_manifest("physics")


def test_export_requires_existing_manifest(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    repo.prepare_store("physics")
    with pytest.raises(vector_store.VectorStoreError):
        repo.export_store("physics", tmp_path / "out.zip")


def test_manifest_round_trip_preserves_documents(tmp_path):
    manifest = _dummy_manifest("physics")
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    repo.ensure_root()
    path = repo.prepare_store("physics") / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    loaded = repo.load_manifest("physics")
    assert loaded.documents[0].source_path == "doc1.md"


def test_list_manifests_skips_non_directories(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    (root / "random.txt").write_text("x", encoding="utf-8")
    assert repo.list_manifests() == []


def test_import_requires_manifest(tmp_path):
    archive = tmp_path / "missing_manifest.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("records.json", "{}")
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    repo.ensure_root()
    with pytest.raises(vector_store.VectorStoreError):
        repo.import_store("physics", archive)


def test_load_manifest_detects_name_mismatch(tmp_path):
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    repo.ensure_root()
    store_dir = repo.prepare_store("physics")
    wrong_manifest = _dummy_manifest("chemistry")
    (store_dir / "manifest.json").write_text(
        json.dumps(wrong_manifest.to_dict()),
        encoding="utf-8",
    )
    with pytest.raises(vector_store.VectorStoreError):
        repo.load_manifest("physics")


def test_inmemory_backend_validates_lengths(tmp_path):
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        backend.create(
            tmp_path,
            texts=["a"],
            embeddings=[[0.0], [1.0]],
            metadatas=[{"i": 1}],
        )


def test_faiss_backend_with_stubs(tmp_path, monkeypatch):
    backend = vector_store.FaissVectorStoreBackend()

    class FakeIndex:
        def __init__(self, dim):
            self.dim = dim
            self.seen = None

        def add(self, array):
            self.seen = list(array)

    fake_numpy = types.SimpleNamespace(array=lambda data, dtype=None: data)
    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=lambda dim: FakeIndex(dim),
        write_index=lambda index, path: path,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    backend.create(
        tmp_path,
        texts=["chunk"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"source_path": "doc", "chunk_index": 0}],
    )
    records = json.loads(
        (tmp_path / "records.json").read_text(encoding="utf-8")
    )
    assert records[0]["text"] == "chunk"


def test_safe_extract_sets_permissions(tmp_path):
    archive = tmp_path / "archive.zip"
    subdir = "dir/file.txt"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dir/", "")
        zf.writestr(subdir, "data")
    destination = tmp_path / "dest"
    with zipfile.ZipFile(archive, "r") as zf:
        vector_store._safe_extract(zf, destination)
    assert (destination / subdir).exists()


def test_faiss_backend_errors(monkeypatch, tmp_path):
    backend = vector_store.FaissVectorStoreBackend()
    fake_numpy = types.SimpleNamespace(array=lambda data, dtype=None: data)
    fake_faiss = types.SimpleNamespace(
        IndexFlatL2=lambda dim: types.SimpleNamespace(add=lambda data: None),
        write_index=lambda index, path: path,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    with pytest.raises(vector_store.VectorStoreError):
        backend.create(tmp_path, texts=["a"], embeddings=[], metadatas=[])
    with pytest.raises(vector_store.VectorStoreError):
        backend.create(
            tmp_path,
            texts=["a"],
            embeddings=[[0.0]],
            metadatas=[{}, {}],
        )
    with pytest.raises(vector_store.VectorStoreError):
        backend.create(
            tmp_path,
            texts=["a"],
            embeddings=[[]],
            metadatas=[{}],
        )
    with pytest.raises(vector_store.VectorStoreError):
        backend.create(
            tmp_path,
            texts=["a", "b"],
            embeddings=[[0.0], [1.0, 2.0]],
            metadatas=[{}, {}],
        )


def test_repository_properties_and_errors(tmp_path):
    repo = vector_store.VectorStoreRepository(tmp_path / "root")
    repo.ensure_root()
    assert repo.root == tmp_path / "root"
    repo.prepare_store("physics")
    with pytest.raises(vector_store.VectorStoreError):
        repo.prepare_store("physics")
    repo.prepare_store("physics", overwrite=True)
    with pytest.raises(vector_store.VectorStoreError):
        repo.load_manifest("physics")


def test_list_manifests_handles_invalid_manifest(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    store_dir = repo.prepare_store("physics")
    (store_dir / "manifest.json").write_text("not json", encoding="utf-8")
    assert repo.list_manifests() == []


def test_list_manifests_skips_missing_manifest(tmp_path):
    root = tmp_path / "root"
    repo = vector_store.VectorStoreRepository(root)
    repo.ensure_root()
    repo.prepare_store("physics")
    assert repo.list_manifests() == []
