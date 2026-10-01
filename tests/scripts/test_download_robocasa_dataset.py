# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from huggingface_hub import HfApi, configure_http_backend
from huggingface_hub.errors import HfHubHTTPError
import pytest
from scripts import download_robocasa_dataset as downloader
from urllib3.response import HTTPResponse


@pytest.mark.parametrize(
    ("headers", "delay"),
    [
        ({"RateLimit": '"api";r=0;t=40'}, 41),
        ({"Retry-After": "60"}, 61),
        ({"Retry-After": "60", "RateLimit": '"api";r=0;t=90'}, 91),
        ({"RateLimit": '"api";r=0;t=0'}, 1),
        ({}, 301),
        ({"Retry-After": "invalid", "RateLimit": "invalid"}, 301),
    ],
)
def test_rate_limit_delay_honors_server_reset(headers, delay):
    response = HTTPResponse(status=429, headers=headers)
    assert downloader.HubRateLimitRetry().get_retry_after(response) == delay


@pytest.fixture
def local_hub(monkeypatch):
    """Exercise the installed HF HTTP stack without contacting the actual Hub."""
    responses = []
    calls = []
    sleeps = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            status, headers, payload = responses.pop(0)
            body = json.dumps(payload).encode()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    configure_http_backend(backend_factory=downloader.create_http_session)
    try:
        api = HfApi(endpoint=f"http://127.0.0.1:{server.server_port}", token=False)
        yield api, responses, calls, sleeps
    finally:
        configure_http_backend()
        server.shutdown()
        server.server_close()
        thread.join()


def test_hub_api_waits_then_retries_same_request(local_hub):
    api, responses, calls, sleeps = local_hub
    responses.extend(
        [
            (429, {"RateLimit": '"api";r=0;t=12'}, {"error": "Too many requests"}),
            (200, {}, {"id": "test/data", "sha": "a" * 40}),
        ]
    )
    assert api.dataset_info("test/data", expand=["sha"]).sha == "a" * 40
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert sleeps == [13]


def test_rate_limit_retries_are_bounded(local_hub):
    api, responses, calls, sleeps = local_hub
    responses.extend([(429, {"Retry-After": "0"}, {"error": "Too many requests"})] * 9)
    with pytest.raises(HfHubHTTPError) as error:
        api.dataset_info("test/data", expand=["sha"])
    assert error.value.response.status_code == 429
    assert len(calls) == 9
    assert sleeps == [1] * 8


def test_auth_errors_are_not_retried(local_hub):
    api, responses, calls, sleeps = local_hub
    responses.append((401, {}, {"error": "Unauthorized"}))
    with pytest.raises(HfHubHTTPError):
        api.dataset_info("test/data", expand=["sha"])
    assert len(calls) == 1
    assert not sleeps


@pytest.mark.parametrize("max_workers", [1, 2])
def test_download_failure_does_not_start_later_batches(max_workers):
    calls = []

    def fail_download(path):
        calls.append(path)
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError, match="permanent failure"):
        downloader.download_files(fail_download, [str(i) for i in range(10)], max_workers, "test")
    assert set(calls).issubset({str(i) for i in range(max_workers)})
