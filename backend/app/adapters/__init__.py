"""External adapters: Elastic, Milvus, PE/GLAP, NVILA QA, and Gemini grounding.

Each adapter supports a `mock_mode` that returns deterministic fixtures so the
frontend can be developed and tests can run without live services. In production
mode adapters never fabricate results — they raise/propagate errors instead.
"""
