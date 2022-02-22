# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-mongo", action="store_true", default=False, help="Do not run mongodb tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-mongo"):
        skip_mongo = pytest.mark.skip(reason="--no-mongo option given, skipping...")
        for item in items:
            if "nomongo" in item.keywords:
                item.add_marker(skip_mongo)

