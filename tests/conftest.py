import importlib.util
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "_autocaption_under_test"


def load_source_module(module_name: str, relative_path: str):
    """Load a project module directly from its file without executing autocaption/__init__.py."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_package_module(module_name: str):
    """
    Load autocaption/<module_name>.py as a submodule of a stub package, so relative
    imports (e.g. ``from .confidence import ...``) work without executing
    autocaption/__init__.py and loading every model dependency.
    """
    if TEST_PACKAGE not in sys.modules:
        package = types.ModuleType(TEST_PACKAGE)
        package.__path__ = [str(PROJECT_ROOT / "autocaption")]
        sys.modules[TEST_PACKAGE] = package
    full_name = f"{TEST_PACKAGE}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    return load_source_module(
        f"{TEST_PACKAGE}.{module_name}",
        f"autocaption/{module_name}.py",
    )


@pytest.fixture(scope="session")
def confidence_module():
    return load_package_module("confidence")


@pytest.fixture(scope="session")
def image_loader_module():
    return load_source_module(
        "_image_loader_under_test",
        "autocaption/image_loader.py",
    )


@pytest.fixture(scope="session")
def image_processor_module():
    return load_package_module("image_processor")


@pytest.fixture(scope="session")
def feature_extractor_module():
    return load_package_module("feature_extractor")
