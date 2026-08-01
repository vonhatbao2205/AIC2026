"""DeepSeek web-search grounding for QA knowledge completion.

NVILA remains responsible for locating visual evidence. This adapter receives
only the question, canonical candidate IDs, and compact OCR/ASR/audio clues. It
uses DeepSeek's server-side `web_search` tool to resolve missing world knowledge
and returns answer alternatives with citations; it never selects a DRES frame by
itself.

Transport note: the built-in `web_search` tool exists only on the **Responses**
API (`POST /responses`) and only for `deepseek-v4-flash`. `/chat/completions`
rejects every non-`function` tool type, so this adapter must not be ported to it.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from ..config import Settings


class WebGroundingUnavailable(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    tagged = re.search(r"<grounded_json>\s*(\{.*?\})\s*</grounded_json>", text, re.I | re.S)
    raw = tagged.group(1) if tagged else text
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("DeepSeek grounded response did not contain a JSON object")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("DeepSeek grounded JSON must be an object")
    return data


def _clean_source_url(url: str) -> str:
    """Drop the `#ws_call_id=…` fragment DeepSeek appends to every opened page."""
    return re.sub(r"#ws_call_id=[^#]*$", "", str(url or "").strip())


def _parse_response(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten the Responses API output array.

    Items are `reasoning`, `web_search_call` and `message`. Reasoning text is
    deliberately dropped: it is the model's private trace, not an answer or a
    citation. Search actions carry either the issued `queries` or the `url` of a
    page the server actually opened — the latter is what we can cite.
    """
    text_parts: list[str] = []
    queries: list[str] = []
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
        elif kind == "web_search_call":
            action = item.get("action")
            if not isinstance(action, dict):
                continue
            for query in action.get("queries") or []:
                query = str(query)
                # The server mixes its internal call id into the query list.
                if query.startswith("ws_call_id=") or query in queries:
                    continue
                queries.append(query[:500])
            url = _clean_source_url(action.get("url") or "")
            if url.startswith(("https://", "http://")) and url not in seen_urls:
                seen_urls.add(url)
                host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
                sources.append({"title": host[:300], "url": url[:2000]})

    return {
        "text": "\n".join(text_parts).strip(),
        "sources": sources[:12],
        "queries": queries[:8],
        "search_suggestions_html": "",  # no DeepSeek equivalent of Google's widget
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
    return f"""You are the web-search grounding stage of a competitive video QA system.

The video system already retrieved frames and extracted noisy clues. Use web search only to resolve missing external knowledge, aliases, canonical names, people, brands, places, organizations, or factual relationships. Video evidence remains primary: do not invent an unrelated answer and do not choose a frame that lacks a relevant clue.

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
- When the question asks for the NAME of an entity, the short common form and the fuller official form are genuinely different answers under a strict grader. List each as its own alternative, formatted as the question demands, ordered by which a Vietnamese broadcast audience would most likely intend (e.g. for the Disney castle logo: "DISNEY", "WALTDISNEY", "THEWALTDISNEYCOMPANY").
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


class DeepSeekGroundingClient:
    def __init__(self, settings: Settings):
        self.s = settings

    async def health(self) -> dict[str, Any]:
        if self.s.mock_mode:
            return {"ok": True, "mode": "mock", "model": "mock-deepseek-search"}
        if not self.s.has_web_grounding:
            return {"ok": False, "mode": "disabled", "error": "DEEPSEEK_API_KEY is not configured"}
        # Avoid a billable search in /api/health. A real request reports upstream
        # errors as warnings without taking down NVILA or core retrieval.
        return {
            "ok": True,
            "mode": "configured",
            "model": self.s.deepseek_grounding_model,
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
        if not self.s.has_web_grounding:
            raise WebGroundingUnavailable("DeepSeek web grounding is not configured")

        started = time.perf_counter()
        model = self.s.deepseek_grounding_model
        payload = {
            "model": model,
            "input": _grounding_prompt(question, candidates, visual_analysis, max_answers),
            "tools": [{"type": "web_search"}],
            # Thinking is on by default and shares this budget with the answer;
            # too small a cap returns reasoning only and an empty message.
            "max_output_tokens": self.s.deepseek_grounding_max_output_tokens,
            "reasoning": {"effort": self.s.deepseek_grounding_reasoning_effort},
        }
        try:
            timeout = httpx.Timeout(self.s.deepseek_grounding_timeout_seconds, connect=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.s.deepseek_grounding_base_url}/responses",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.s.deepseek_api_key or ''}",
                    },
                )
                response.raise_for_status()
                raw = response.json()
        except httpx.HTTPStatusError as exc:
            raise WebGroundingUnavailable(
                f"DeepSeek grounding returned {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalized at service boundary
            raise WebGroundingUnavailable(f"DeepSeek grounding unreachable: {exc}") from exc

        parsed = _parse_response(raw)
        if raw.get("status") == "incomplete":
            reason = ((raw.get("incomplete_details") or {}).get("reason")) or "unknown"
            if not parsed["text"]:
                raise WebGroundingUnavailable(
                    f"DeepSeek grounding stopped early ({reason}) before writing an answer; "
                    "raise DEEPSEEK_GROUNDING_MAX_OUTPUT_TOKENS."
                )
        try:
            grounded = _extract_json_object(parsed["text"])
        except (ValueError, json.JSONDecodeError) as exc:
            raise WebGroundingUnavailable(f"DeepSeek grounding output was not parseable: {exc}") from exc
        grounded.update({
            "model": model,
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
        "model": "mock-deepseek-search",
        "sources": [{"title": "example.test", "url": "https://example.test/disney"}],
        "queries": ["Disney canonical name"],
        "search_suggestions_html": "",
        "summary": "Mock grounded response.",
        "latency_ms": 1.0,
    }
