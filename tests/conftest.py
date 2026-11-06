import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: fits real models (statsmodels, NumPyro); slower"
    )
