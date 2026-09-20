"""Request-scoped work counters, also propagated by asyncio.to_thread."""
from contextvars import ContextVar

work: ContextVar[dict | None] = ContextVar("retrieval_work", default=None)


def count(name: str, amount: int = 1):
    counters = work.get()
    if counters is not None:
        counters[name] = counters.get(name, 0) + amount
