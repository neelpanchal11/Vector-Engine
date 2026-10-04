from scripts.build_release_bundle import build_bundle


def test_build_release_bundle_manifest(tmp_path):
    payload = build_bundle(str(tmp_path))
    assert payload["bundle_version"] == "v1.2.0"
    assert "ready_for_submission" in payload
    assert "missing_paths" in payload
    assert "docs/releases/v1.2.0.md" in [item["path"] for item in payload["documents"]]
    assert "artifacts/ivf_benchmark/ivf_batching_comparison.json" in [
        item["path"] for item in payload["artifacts"]
    ]
    assert (tmp_path / "release_bundle_manifest.v1.json").exists()
