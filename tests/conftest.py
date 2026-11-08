import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory) -> str:
    from make_fixture import build

    path = str(tmp_path_factory.mktemp("pdf") / "fixture.pdf")
    build(path)
    return path


@pytest.fixture(scope="session")
def real_pdf() -> str:
    """A real EP amendments PDF, if the developer points EP_SAMPLE_PDF at one."""
    path = os.getenv("EP_SAMPLE_PDF")
    if not path or not os.path.exists(path):
        pytest.skip("Set EP_SAMPLE_PDF to run the real-document tests")
    return path
