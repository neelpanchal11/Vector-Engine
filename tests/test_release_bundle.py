from scripts.build_release_bundle import build_bundle


def test_build_release_bundle_manifest(tmp_path):
    payload = build_bundle(str(tmp_path))
    assert "ready_for_submission" in payload
    assert "missing_paths" in payload
    assert (tmp_path / "release_bundle_manifest.v1.json").exists()
