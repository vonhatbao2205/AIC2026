import { useEffect, useRef, useState } from "react";
import chatGptLogo from "../../../chatgpt-seeklogo.png";

const VERBS = [
  "Pondering",
  "Ruminating",
  "Mulling",
  "Thinking",
  "Perusing",
  "Synthesizing",
  "Ideating",
  "Wrangling",
  "Percolating",
  "Connecting",
] as const;

const SUBLINES = [
  { afterSeconds: 0, text: "Looking at the clues…" },
  { afterSeconds: 4, text: "Working through it…" },
  { afterSeconds: 9, text: "Connecting a few possibilities…" },
  { afterSeconds: 15, text: "Weighing the candidates…" },
  { afterSeconds: 23, text: "Synthesizing what fits best…" },
  { afterSeconds: 32, text: "Taking a closer look…" },
] as const;

function sublineAt(elapsedSeconds: number): string {
  for (let index = SUBLINES.length - 1; index >= 0; index -= 1) {
    if (elapsedSeconds >= SUBLINES[index].afterSeconds) return SUBLINES[index].text;
  }
  return SUBLINES[0].text;
}

/** Decorative search feedback inspired by Claude Code's spinner UX. The copy
 *  deliberately describes no internal stage: retrieval remains the source of
 *  truth, while this component only makes a long wait feel alive. */
export function SearchThinking() {
  const startedAt = useRef(Date.now());
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [verbIndex, setVerbIndex] = useState(0);

  useEffect(() => {
    const elapsedTimer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt.current) / 1_000));
    }, 250);
    const verbTimer = window.setInterval(() => {
      setVerbIndex((current) => {
        // Always move at least one slot so two consecutive labels never match.
        const offset = 1 + Math.floor(Math.random() * (VERBS.length - 1));
        return (current + offset) % VERBS.length;
      });
    }, 1_800);

    return () => {
      window.clearInterval(elapsedTimer);
      window.clearInterval(verbTimer);
    };
  }, []);

  const subline = sublineAt(elapsedSeconds);

  return (
    <div className="search-thinking" role="status" aria-label="Đang tìm kiếm">
      <div className="search-thinking-main" aria-hidden="true">
        <span className="search-thinking-logo" data-testid="search-thinking-logo">
          <img src={chatGptLogo} alt="" draggable={false} />
        </span>
        <span key={verbIndex} className="search-thinking-verb" data-testid="search-thinking-verb">
          {VERBS[verbIndex]}…
        </span>
        <span className="search-thinking-time mono" data-testid="search-thinking-time">
          {elapsedSeconds}s
        </span>
      </div>
      <div key={subline} className="search-thinking-subline" data-testid="search-thinking-subline" aria-hidden="true">
        {subline}
      </div>
    </div>
  );
}
