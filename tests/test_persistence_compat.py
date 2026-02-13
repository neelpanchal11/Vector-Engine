import json

import numpy as np
import pytest

from vector_engine import VectorArray, VectorIndex
from vector_engine.io import load_manifest


def test_manifest_version_rejected(tmp_path):
    manifest = {
        "version": "0.0",
        "backend": "bruteforce",
        "metric_name": "l2",
        "higher_is_better": False,
        "dim": 4,
        "count": 2,
        "backend_config": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_error"):
        load_manifest(str(tmp_path))


def test_ids_checksum_validation(tmp_path):
    xb = VectorArray.from_numpy(np.random.randn(15, 6).astype(np.float32), ids=np.arange(15))
    index = VectorIndex.create(xb, metric="l2", backend="bruteforce")
    save_dir = tmp_path / "idx"
    index.save(str(save_dir))

    ids_path = save_dir / "ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    ids[0] = "tampered-id"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")

    with pytest.raises(ValueError, match="index_error"):
        VectorIndex.load(str(save_dir))
