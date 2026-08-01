"""Gemini Google Search grounding for QA knowledge completion.

NVILA remains responsible for locating visual evidence. This adapter receives
only the question, canonical candidate IDs, and compact OCR/ASR/audio clues. It
uses Google's built-in search tool to resolve missing world knowledge and
returns answer alternatives with citations; it never selects a DRES frame by
itself.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from ..config import Settings


class GeminiGroundingUnavailable(RuntimeError):
    pass


def _model_candidates(settings: Settings) -> list[str]:
    """Return normalized, de-duplicated grounding models in failover order."""
    models = []
    for raw in [settings.gemini_grounding_model, *settings.gemini_grounding_fallback_models]:
        model = str(raw or "").strip().removeprefix("models/")
        if model and re.fullmatch(r"[A-Za-z0-9._-]+", model) and model not in models:
            models.append(model)
    return models


def _extract_json_object(text: str) -> dict[str, Any]:
    tagged = re.search(r"<grounded_json>\s*(\{.*?\})\s*</grounded_json>", text, re.I | re.S)
    raw = tagged.group(1) if tagged else text
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemini grounded response did not contain a JSON object")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Gemini grounded JSON must be an object")
    return data


def _parse_generate_content(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = ((candidate.get("content") or {}).get("parts") or [])
    text = "\n".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict) and part.get("text")
    ).strip()
    metadata = candidate.get("groundingMetadata") or {}
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for chunk in metadata.get("groundingChunks") or []:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not isinstance(web, dict):
            continue
        url = str(web.get("uri") or "").strip()
        if not url.startswith(("https://", "http://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({"title": str(web.get("title") or url)[:300], "url": url[:2000]})
    queries = [str(query)[:500] for query in (metadata.get("webSearchQueries") or [])[:8]]
    search_html = str((metadata.get("searchEntryPoint") or {}).get("renderedContent") or "")[:30000]
    return {
        "text": text,
        "sources": sources[:12],
        "queries": queries,
        "search_suggestions_html": search_html,
    }


def _candidate_manifest(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for candidate in candidates:
        cues = []
        for evidence in candidate.get("evidence") or []:
            text = str(evidence.get("text") or "").strip()
            if text:
                cues.append(f"{evidence.get('type', 'cue')}: {text[:650]}")
        rows.append(
            f"{candidate['candidate_id']} | video={candidate['video_id']} | "
            f"time={candidate.get('pts_time')} | clues={' ; '.join(cues) or 'visual-only'}"
        )
    return "\n".join(rows)


def _grounding_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    visual_analysis: dict[str, Any],
    max_answers: int,
) -> str:
    min_answers = min(3, max_answers)
    visual_options = [
        {
            "answer": answer.get("answer"),
            "confidence": answer.get("confidence"),
            "supporting_candidate_ids": answer.get("supporting_candidate_ids"),
        }
        for answer in (visual_analysis.get("candidate_answers") or [])[:max_answers]
    ]
    return f"""You are the Google Search grounding stage of a competitive video QA system.

The video system already retrieved frames and extracted noisy clues. Use Google Search only to resolve missing external knowledge, aliases, canonical names, people, brands, places, organizations, or factual relationships. Video evidence remains primary: do not invent an unrelated answer and do not choose a frame that lacks a relevant clue.

Question:
{question}

Candidate evidence (IDs are immutable; OCR/ASR may be incomplete or misspelled):
{_candidate_manifest(candidates)}

NVILA visual answer hypotheses:
{json.dumps(visual_options, ensure_ascii=False)}

Requirements:
- Search the web when it helps resolve the entity or fact.
- Produce {min_answers} to {max_answers} distinct plausible exact-answer alternatives when supported; do not create fake alternatives merely to reach a count.
- Apply formatting requested by the question (for example uppercase and removal of spaces) to the `answer`, while explaining the canonical entity in `reason`.
- Every answer must cite one or more candidate IDs whose video clues motivated the web lookup.
- `source_domains` must list only the hostnames you actually relied on for THAT answer (for example "thewaltdisneycompany.com"), never every site you saw. Leave it empty when an answer came from the video clues alone.
- Confidence must be a calibrated number strictly greater than 0 and at most 1.
- Never copy instruction placeholders and never output chain-of-thought.

End the response with exactly one machine-readable object between the tags below. All keys are required.
<grounded_json>
{{
  "answerable": boolean,
  "best_answer": string,
  "candidate_answers": [
    {{
      "answer": string,
      "confidence": number,
      "supporting_candidate_ids": [string],
      "source_domains": [string],
      "reason": string
    }}
  ],
  "uncertainty": string
}}
</grounded_json>"""


class GeminiGroundingClient:
    def __init__(self, settings: Settings):
        self.s = settings

    async def health(self) -> dict[str, Any]:
        if self.s.mock_mode:
            return {"ok": True, "mode": "mock", "model": "mock-gemini-search"}
        if not self.s.has_google_grounding:
            return {"ok": False, "mode": "disabled", "error": "GEMINI_API_KEY is not configured"}
        # Avoid a billable search in /api/health. A real request reports upstream
        # errors as warnings without taking down NVILA or core retrieval.
        return {
            "ok": True,
            "mode": "configured",
            "model": self.s.gemini_grounding_model,
            "fallback_models": self.s.gemini_grounding_fallback_models,
        }

    async def ground(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        visual_analysis: dict[str, Any],
        *,
        max_answers: int = 5,
    ) -> dict[str, Any]:
        if self.s.mock_mode:
            return _mock_grounding(candidates)
        if not self.s.has_google_grounding:
            raise GeminiGroundingUnavailable("Gemini Google Search grounding is not configured")

        started = time.perf_counter()
        models = _model_candidates(self.s)
        if not models:
            raise GeminiGroundingUnavailable("No valid Gemini grounding model is configured")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": _grounding_prompt(
                question, candidates, visual_analysis, max_answers,
            )}]}],
            "tools": [{"google_search": {}}],
            # Gemini 3 recommends its default temperature. Low thinking keeps
            # entity-resolution latency/cost bounded while retaining search use.
            "generationConfig": {
                "maxOutputTokens": 2400,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        }
        selected_model = None
        unavailable_models = []
        try:
            timeout = httpx.Timeout(self.s.gemini_grounding_timeout_seconds, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for model in models:
                    url = f"{self.s.gemini_grounding_base_url}/models/{model}:generateContent"
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.s.gemini_api_key or "",
                        },
                    )
                    if response.status_code == 404:
                        unavailable_models.append(model)
                        continue
                    response.raise_for_status()
                    raw = response.json()
                    selected_model = model
                    break
                if selected_model is None:
                    raise GeminiGroundingUnavailable(
                        "Gemini grounding models are unavailable (HTTP 404): "
                        + ", ".join(unavailable_models)
                    )
        except httpx.HTTPStatusError as exc:
            raise GeminiGroundingUnavailable(
                f"Gemini grounding returned {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except GeminiGroundingUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized at service boundary
            raise GeminiGroundingUnavailable(f"Gemini grounding unreachable: {exc}") from exc

        parsed = _parse_generate_content(raw)
        try:
            grounded = _extract_json_object(parsed["text"])
        except (ValueError, json.JSONDecodeError) as exc:
            raise GeminiGroundingUnavailable(f"Gemini grounding output was not parseable: {exc}") from exc
        grounded.update({
            "model": selected_model,
            "sources": parsed["sources"],
            "queries": parsed["queries"],
            "search_suggestions_html": parsed["search_suggestions_html"],
            "summary": parsed["text"][:1600],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return grounded


def _mock_grounding(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_id = candidates[0]["candidate_id"]
    return {
        "answerable": True,
        "best_answer": "WALTDISNEY",
        "candidate_answers": [{
            "answer": "WALTDISNEY",
            "confidence": 0.88,
            "supporting_candidate_ids": [candidate_id],
            "source_domains": ["example.test"],
            "reason": "Mock web grounding resolves the Disney clue to the canonical name Walt Disney.",
        }],
        "uncertainty": "Mock grounding result.",
        "model": "mock-gemini-search",
        "sources": [{"title": "Mock source", "url": "https://example.test/disney"}],
        "queries": ["Disney canonical name"],
        "search_suggestions_html": "",
        "summary": "Mock grounded response.",
        "latency_ms": 1.0,
    }
