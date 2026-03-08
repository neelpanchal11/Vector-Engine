from scripts import env_diagnostics


def test_collect_env_report_contains_expected_keys(monkeypatch):
    monkeypatch.setattr(
        env_diagnostics,
        "_numpy_info_subprocess",
        lambda: {"returncode": 0, "stdout": "1.26.4\nblas=ok", "stderr": "", "segfault_suspected": False},
    )
    monkeypatch.setattr(
        env_diagnostics,
        "_numpy_smoke_subprocess",
        lambda: {"returncode": 0, "stdout": "ok", "stderr": "", "segfault_suspected": False},
    )
    payload = env_diagnostics.collect_env_report()
    assert set(payload.keys()) == {"python", "platform", "numpy", "suggestions"}
    assert payload["numpy"]["info_probe"]["segfault_suspected"] is False
    assert payload["numpy"]["smoke"]["segfault_suspected"] is False
