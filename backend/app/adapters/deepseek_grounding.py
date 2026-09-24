"""DeepSeek web-search grounding for QA knowledge completion.

NVILA remains responsible for locating visual evidence. This adapter receives
only the question, canonical candidate IDs, and compact OCR/ASR/audio clues. It
uses DeepSeek's server-side `web_search` tool to resolve missing world knowledge
and returns answer alternatives with citations; it never selects a DRES frame by
itself.

Transport note: the built-in `web_search` tool exists only on the **Responses**
API (`POST /responses`); `/chat/completions` rejects every non-`function` tool
type, so this adapter must not be ported to it. Measured 2026-09-25: only
`deepseek-v4-pro` actually runs the tool. `deepseek-flash` (V4.1 Flash, which the
old id `deepseek-v4-flash` now resolves to) accepts it but writes fake
`<tool_call>` text instead of searching; `_parse_response` detects that so a
search that never happened can never be passed off as "web grounded".
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
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        # The model hand-writes this object and sometimes leaves a trailing comma.
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw[start : end + 1]))
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

    searched = any(isinstance(item, dict) and item.get("type") == "web_search_call" for item in data.get("output") or [])
    text = "\n".join(text_parts).strip()
    return {
        "text": text,
        "sources": sources[:12],
        "queries": queries[:8],
        "searched": searched,
        # A model that cannot run the server-side tool writes one as text.
        "fake_tool_call": not searched and bool(re.search(r"<tool_call>|<tool_name>|\"name\":\s*\"search\"", text)),
        "search_suggestions_html": "",  # no DeepSeek equivalent of Google's widget
    }


# Sent as the Responses API `instructions` field. It carries the standing role and
# grading contract; the per-request prompt carries the question and candidates.
_SYSTEM_INSTRUCTIONS = """You are the external-knowledge stage of a Vietnamese video question-answering system in the AIC 2026 final round, graded as an exact string under a clock, where every wrong submission costs points.

The collection: HTV "60 giây" news bulletins (K01-K20, L21-L22, and M01-M10 from 2026, a few narrated in English), road cycling (L23, and S01: the 2026 HTV Television Cup, 12 stages), lion dance (L24), exam revision lectures (L25), cooking (L26), Vietnamese culture and the Mekong delta (L27-L29), user shorts about charity (L30), and fixed traffic cameras at Ho Chi Minh City junctions in June 2026 (N001-N100).

Operating rules:
- The frames were already inspected by a vision model; its findings and the OCR/ASR clues are your starting point. Your job is what the frames cannot show: proper names, aliases, brands, organisations, places, people, dates, event results and the relationships between them.
- Search whenever the answer hinges on a named real-world entity or a fact outside the frames, and when a clue looks partial, misspelled or transliterated: Vietnamese ASR routinely mangles foreign names, so search the corrected form. Search in Vietnamese for Vietnamese subjects and in English for international ones.
- Vietnamese provinces and communes were merged in 2025. "tại thời điểm đó" (at that time) asks for the name in force when the video was made, not today's.
- Apply every format constraint of the question (uppercase, English, no spaces, no diacritics, digits only) inside the `answer` field itself, never only in the explanation.
- Answer with the name itself, without the generic word the question already supplies ("which virus" -> Marburg, not "Marburg virus"; "which commune" -> Giang Ly, not "Xã Giang Ly").
- A real-world entity often has several legitimate names (short brand, full corporate name, local Vietnamese name, e.g. "DISNEY", "WALTDISNEY", "THEWALTDISNEYCOMPANY"). Under an exact-string grader these are different answers: return each as its own alternative, ordered by which one a Vietnamese broadcaster most likely means.
- Explain the canonical entity in `reason`, keep `answer` in the exact demanded format, cite only domains you actually opened, and output no chain-of-thought."""


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
            "reason": str(answer.get("reason") or "")[:300],
        }
        for answer in (visual_analysis.get("candidate_answers") or [])[:max_answers]
    ]
    # What the vision pass SAW in each frame is the best search seed there is.
    frame_findings = [
        {"candidate_id": hotspot.get("candidate_id"), "seen": str(hotspot.get("answer_support") or "")[:300]}
        for hotspot in (visual_analysis.get("hotspots") or [])[:5]
    ]
    return f"""Resolve the external knowledge this question needs. Video evidence remains primary: do not invent an unrelated answer and do not tie an answer to a frame that lacks a relevant clue.

Question:
{question}

Candidate evidence (IDs are immutable; OCR/ASR may be incomplete or misspelled):
{_candidate_manifest(candidates)}

What the vision model saw in the most relevant frames:
{json.dumps(frame_findings, ensure_ascii=False)}

Vision model answer hypotheses (they may be right, partial or wrongly formatted):
{json.dumps(visual_options, ensure_ascii=False)}

Requirements for this request:
- Produce {min_answers} to {max_answers} distinct plausible exact-answer alternatives when supported; do not create fake alternatives merely to reach a count.
- Every answer must cite one or more candidate IDs whose video clues motivated the web lookup.
- `source_domains` must list only the hostnames you actually relied on for THAT answer, never every site you saw. Leave it empty when an answer came from the video clues alone.
- Confidence must be a calibrated number strictly greater than 0 and at most 1.
- Never copy instruction placeholders.

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
            "instructions": _SYSTEM_INSTRUCTIONS,
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
        if parsed["fake_tool_call"]:
            raise WebGroundingUnavailable(
                f"{model} did not run a web search (it wrote the tool call as text). DeepSeek's "
                "built-in web search currently runs only on deepseek-v4-pro; set DEEPSEEK_GROUNDING_MODEL."
            )
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
            "searched": parsed["searched"],
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
