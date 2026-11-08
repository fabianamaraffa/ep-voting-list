"""Starts the real Gradio app (with fake models) and drives it like a browser user would."""

import importlib
import os
import sys

import pytest

gradio_client = pytest.importorskip("gradio_client")


@pytest.fixture(scope="module")
def server():
    os.environ["EP_USE_STUBS"] = "1"
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    _, url, _ = app.demo.queue().launch(prevent_thread_lock=True, server_port=7899, quiet=True)
    yield url
    app.demo.close()


def test_upload_analyse_and_export(server, fixture_pdf):
    client = gradio_client.Client(server, verbose=False)
    table, status = client.predict(
        gradio_client.handle_file(fixture_pdf),
        "Youth Guarantee and youth instruments, paragraph text",
        0.05,
        10,
        api_name="/analyse",
    )
    rows = table["data"]
    assert 1 <= len(rows) <= 6
    assert table["headers"][:5] == ["Amendment", "Authors", "Location", "Vote", "Rationale"]
    assert "analysed" in status

    xlsx = client.predict(table, api_name="/export_table")
    assert str(xlsx).endswith(".xlsx") and os.path.getsize(xlsx) > 0
