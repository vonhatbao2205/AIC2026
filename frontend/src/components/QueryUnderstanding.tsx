import { useState } from "react";
import type { ParsedQuery } from "../api/types";
import { CHANNELS } from "../lib/constants";
import { ChannelBadge } from "./Badges";

// Long text (e.g. an EN visual that echoes a 1000-char query) is clamped to keep
// the side panel compact; click to expand.
function Expandable({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > 140;
  return (
    <span className="v" style={{ display: "block" }}>
      <span className={open || !long ? "" : "clamp-3"}>{text}</span>
      {long && (
        <button className="linklike" onClick={() => setOpen((o) => !o)}>
          {open ? "thu gọn" : "xem thêm"}
        </button>
      )}
    </span>
  );
}

export function QueryUnderstanding({ parsed }: { parsed: ParsedQuery | null }) {
  if (!parsed) return null;
  const enabled = CHANNELS.filter((c) => parsed.channels?.[c]?.enabled);
  return (
    <div className="panel">
      <h3>Query understanding {parsed._engine ? `· ${parsed._engine}` : ""}</h3>
      <div className="kv">
        <span className="k">type</span>
        <span className="v">{parsed.query_type} ({(parsed.confidence * 100).toFixed(0)}%)</span>
        <span className="k">EN visual</span>
        <Expandable text={parsed.translated_en_visual || "—"} />
      </div>
      <div className="row" style={{ marginBottom: 8 }}>
        <span style={{ color: "var(--fg-faint)", fontSize: 12 }}>routed:</span>
        {enabled.length ? enabled.map((c) => <ChannelBadge key={c} channel={c} />) : <span style={{ fontSize: 12 }}>none</span>}
      </div>
      {parsed.filters?.must_not_include?.length ? (
        <div style={{ fontSize: 12 }}>
          <span style={{ color: "var(--bad)" }}>negations:</span>{" "}
          {parsed.filters.must_not_include.join(", ")}
        </div>
      ) : null}
      {parsed.ui_hints?.warning_vi ? (
        <div className="warn-banner" style={{ marginTop: 8, borderRadius: 6, borderBottom: "none", border: "1px solid color-mix(in srgb, var(--warn) 40%, transparent)" }}>
          {parsed.ui_hints.warning_vi}
        </div>
      ) : null}
    </div>
  );
}
