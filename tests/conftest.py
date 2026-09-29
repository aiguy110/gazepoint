import pytest


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path, monkeypatch):
    """Keep the developer's ~/.config/gazepoint/config.toml out of the tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
