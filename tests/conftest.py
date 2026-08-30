import pytest


@pytest.fixture(autouse=True)
def _no_real_gopls(monkeypatch):
    """Unit tests must not spawn a workspace gopls (slow / flaky). Opt in via gopls_factory=."""
    monkeypatch.setattr("app.worker_job.start_gopls", lambda root: None)
