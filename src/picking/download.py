"""Pinned downloads with persistent content checksums and resumable ranges."""
from __future__ import annotations

import concurrent.futures
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import complete_file, emit, verified_file


def fetch(url, destination, attempts=5):
    destination = Path(destination)
    if verified_file(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "data-picking-v2/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                length = response.headers.get("Content-Length")
                if response.status != 200:
                    raise ValueError(f"Expected a complete download, received {response.status}")
                with destination.open("wb") as out:
                    while block := response.read(4 * 1024**2):
                        out.write(block)
                if length is not None and destination.stat().st_size != int(length):
                    raise ValueError("Incomplete download")
                etag = response.headers.get("ETag")
            complete_file(destination, source_url=url, etag=etag)
            emit("downloaded", path=str(destination), bytes=destination.stat().st_size)
            return destination
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise
            if attempt + 1 == attempts:
                raise RuntimeError(f"Download failed: HTTP {error.code} ({destination.name})") from None
        except Exception as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Download failed: {type(error).__name__} ({destination.name})") from None
        time.sleep(min(2 ** attempt, 15))


def fetch_many(items, workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, url, path): path for url, path in items}
        for future in concurrent.futures.as_completed(futures):
            future.result()
