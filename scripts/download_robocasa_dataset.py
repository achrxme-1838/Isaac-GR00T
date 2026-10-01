#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download RoboCasa Panda Omron tasks without listing unrelated dataset subtrees.

Usage:
    python scripts/download_robocasa_dataset.py --local-dir /data/lim/datasets/gr00t-sim

Task files retain their repository-relative paths. Re-running with the same output
directory reuses completed downloads through huggingface_hub's local metadata.
HTTP 429 responses wait for the server's rate-limit reset before retrying. To use
regular HTTP downloads with huggingface_hub 0.36, set HF_HUB_DISABLE_XET=1 before
starting the script.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from pathlib import Path
import re
import time

from huggingface_hub import HfApi, configure_http_backend, constants, hf_hub_download
from huggingface_hub.hf_api import RepoFile, RepoFolder
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.exceptions import InvalidHeader
from urllib3.util.retry import Retry


REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
TASK_PREFIX = "single_panda_gripper."


class HubRateLimitRetry(Retry):
    """Honor Hub RateLimit headers as well as standard Retry-After headers."""

    def get_retry_after(self, response) -> float | None:
        try:
            retry_after = super().get_retry_after(response)
        except InvalidHeader:
            retry_after = None
        if response.status != 429:
            return retry_after

        delays = [retry_after] if retry_after is not None else []
        delays.extend(
            float(seconds)
            for seconds in re.findall(
                r"(?:^|[;,])\s*t\s*=\s*(\d+)", response.headers.get("RateLimit", "")
            )
        )
        # A small margin avoids another request at the exact reset boundary.
        # Without headers, wait out the Hub's five-minute quota window.
        return max(delays, default=300) + 1

    def sleep(self, response=None) -> None:
        if response is not None and response.status == 429:
            delay = self.get_retry_after(response)
            print(f"\nHTTP 429: waiting {delay:.0f}s before retrying...", flush=True)
            time.sleep(delay)
        else:
            super().sleep(response)


def create_http_session() -> requests.Session:
    """Retry inside HTTP requests, including paginated listings and Xet tokens."""
    session = requests.Session()
    retry = HubRateLimitRetry(
        total=8,
        connect=0,
        read=0,
        other=0,
        allowed_methods=frozenset({"GET", "HEAD"}),
        status_forcelist={429},
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download_files(
    download_file: Callable[[str], str], files: list[str], max_workers: int, task: str
) -> None:
    # Run on the main thread by default so Ctrl+C also interrupts retry waits.
    if max_workers == 1:
        for path in tqdm(files, desc=task):
            download_file(path)
        return

    # Keep at most one batch in flight. A terminal error must not leave thousands
    # of queued requests running while the executor waits to shut down.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files), desc=task) as progress:
            for batch in batched(files, max_workers):
                for _ in executor.map(download_file, batch):
                    progress.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    if not constants.HF_HUB_OFFLINE:
        configure_http_backend(backend_factory=create_http_session)
    output_dir = args.local_dir.expanduser()
    api = HfApi()
    print("Resolving dataset revision...", flush=True)
    revision = api.dataset_info(REPO_ID, expand=["sha"], timeout=30).sha
    if not revision:
        raise RuntimeError("The Hub did not return a dataset commit hash.")

    print(f"Listing RoboCasa task folders at {revision}...", flush=True)
    tasks = [
        item.path
        for item in api.list_repo_tree(
            REPO_ID, repo_type="dataset", revision=revision, recursive=False
        )
        if isinstance(item, RepoFolder) and item.path.startswith(TASK_PREFIX)
    ]
    tasks.sort(key=lambda path: (path != f"{TASK_PREFIX}OpenDrawer", path))
    if not tasks:
        raise RuntimeError("No RoboCasa Panda Omron task folders found.")
    print(f"Found {len(tasks)} tasks. Destination: {output_dir}", flush=True)

    download_file = partial(
        hf_hub_download,
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
    )
    for index, task in enumerate(tasks, 1):
        print(f"\n[{index}/{len(tasks)}] Listing files in {task}...", flush=True)
        files = []
        for item in api.list_repo_tree(
            REPO_ID,
            path_in_repo=task,
            repo_type="dataset",
            revision=revision,
            recursive=True,
        ):
            if isinstance(item, RepoFile):
                files.append(item.path)
                if len(files) % 1000 == 0:
                    print(f"  Listed {len(files)} files...", flush=True)

        if not files:
            raise RuntimeError(f"No files found in task folder: {task}")
        print(f"  Downloading {len(files)} files...", flush=True)
        download_files(download_file, files, args.max_workers, task)

    print(f"\nAll RoboCasa tasks downloaded to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
