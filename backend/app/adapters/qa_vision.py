"""The QA copilot's visual backend: DeepSeek V4.1 Flash (default) or the NVILA-8B worker.

Both expose `health`, `analyze` (pass 1) and `verify_grounded` (pass 3) with the
same result shape, so the API layer never needs to know which one answered.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings


class QaVisionUnavailable(RuntimeError):
    """The selected visual QA backend could not answer."""


def build_qa_vision_client(settings: "Settings"):
    if settings.qa_vision_backend == "nvila":
        from .nvila_client import NvilaQaClient

        return NvilaQaClient(settings)
    from .deepseek_vision_qa import DeepSeekVisionQaClient

    return DeepSeekVisionQaClient(settings)
