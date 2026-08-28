"""One pooled httpx client per adapter.

Creating an `AsyncClient` per request re-runs the TLS handshake every time.
Measured against this project's Elastic Cloud serverless endpoint: 765 ms per
call with a fresh client vs 249 ms on a warm one — ~500 ms of pure setup, paid
by every channel of every search, and by each of the timeline's queries.

The client is rebuilt when the running event loop changes, so a test that spins
up its own loop (or a reloaded dev server) never reuses a client bound to a dead
one. Timeouts stay per-request because they differ by call site.
"""
from __future__ import annotations

import asyncio

import httpx


class PooledHttpClient:
    def __init__(self, **client_kwargs):
        self._kwargs = client_kwargs
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._loop is not loop:
            self._client = httpx.AsyncClient(**self._kwargs)
            self._loop = loop
        return self._client


def failure_reason(exc: BaseException) -> str:
    """A health/error reason that is never blank.

    httpx transport errors are raised with no message, so `str(exc)` is `""` and
    a health banner ships as "elastic unreachable: " — the operator is told
    something broke but not what, which during a run is the same as being told
    nothing. Fall back to the exception class, which distinguishes a DNS failure
    from a timeout from a refused connection.
    """
    return str(exc).strip() or type(exc).__name__
