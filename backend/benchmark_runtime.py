"""Measure live retrieval latency without exposing backend credentials.

Usage:
    .venv/bin/python benchmark_runtime.py --requests 25 --warmup 1

The request plan bypasses parsing and translation so each operation measures the
retrieval path itself. It intentionally uses fixed cue-bearing queries to keep
the PE, OCR, speech, and audio channels active across repeated runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from statistics import median
from typing import Any

from app.config import get_settings
from app.services.search_service import SearchService


def _plan(*enabled: str) -> dict[str, Any]:
    active = set(enabled)
    return {
        "channels": {
            "image_pe": {
                "enabled": "image_pe" in active,
                "weight": 1.0,
                "queries_en": ["a television presenter in a studio"],
            },
            "ocr": {
                "enabled": "ocr" in active,
                "weight": 0.8,
                "queries_vi": ["Việt Nam"],
                "queries_folded": ["viet nam"],
                "exact_phrases": [],
            },
            "speech": {
                "enabled": "speech" in active,
                "weight": 0.8,
                "queries_vi": ["Việt Nam"],
                "exact_phrases": [],
            },
            "audio": {
                "enabled": "audio" in active,
                "weight": 0.7,
                "queries_en": ["ringing bell"],
                "sound_labels_en": ["Bell"],
            },
        },
        "filters": {},
        "rerank_policy": {"rrf_k": 60},
    }


def _summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": round(median(ordered), 1),
        "p95_ms": round(ordered[p95_index], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
    }


async def _measure(
    service: SearchService, name: str, channels: tuple[str, ...], *, requests: int, warmup: int
) -> dict[str, Any]:
    plan = _plan(*channels)

    async def run_once() -> float:
        started = time.perf_counter()
        await service.retrieve(plan, top_k=100)
        return (time.perf_counter() - started) * 1000

    for _ in range(warmup):
        await run_once()
    samples = [await run_once() for _ in range(requests)]
    return {"operation": name, "requests": requests, **_summary(samples)}


async def _run(requests: int, warmup: int) -> list[dict[str, Any]]:
    service = SearchService(get_settings())
    checks = await asyncio.gather(
        service.elastic.health(), service.milvus.health(), service.pe.health(), service.glap.health()
    )
    names = ("elastic", "milvus", "pe_encoder", "glap_encoder")
    unavailable = [name for name, result in zip(names, checks, strict=True) if not result.get("ok")]
    if unavailable:
        raise RuntimeError(f"live benchmark requires reachable services: {', '.join(unavailable)}")

    operations = (
        ("PE visual", ("image_pe",)),
        ("OCR", ("ocr",)),
        ("Speech", ("speech",)),
        ("Audio", ("audio",)),
        ("Full hybrid", ("image_pe", "ocr", "speech", "audio")),
    )
    return [
        await _measure(service, name, channels, requests=requests, warmup=warmup)
        for name, channels in operations
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=25, help="measured requests per operation")
    parser.add_argument("--warmup", type=int, default=1, help="unreported warm-up requests per operation")
    args = parser.parse_args()
    if args.requests < 2 or args.warmup < 0:
        parser.error("--requests must be at least 2 and --warmup must be non-negative")

    results = asyncio.run(_run(args.requests, args.warmup))
    print(json.dumps({"warmup": args.warmup, "operations": results}, indent=2))


if __name__ == "__main__":
    main()
