import { useState } from "react";
import type { ParsedQuery, TrafficFilterInfo, TrafficFilterMode } from "../api/types";
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
          {open ? "collapse" : "show more"}
        </button>
      )}
    </span>
  );
}

/** What the query's junction / date / time / race-stage cues narrowed, and the
 *  switch to search the cameras and the race without them. Like the scope, the
 *  switch is the policy for the NEXT search; the chips are what this one did. */
function TrafficFilter({
  traffic,
  mode,
  onMode,
}: {
  traffic: TrafficFilterInfo;
  mode: TrafficFilterMode;
  onMode: (mode: TrafficFilterMode) => void;
}) {
  const cameras = traffic.cameras;
  return (
    <div className="traffic-filter" data-testid="traffic-filter">
      <div className="traffic-head">
        <span style={{ color: "var(--fg-faint)" }}>camera / race filter</span>
        <button
          className="btn sm ghost"
          data-testid="traffic-toggle"
          onClick={() => onMode(mode === "off" ? "auto" : "off")}
          title={
            mode === "off"
              ? "Narrow the traffic cameras and the S01 race to the junction, date, time or stage the query names"
              : "Search the traffic cameras and the S01 race without these cues"
          }
        >
          {mode === "off" ? "Apply" : "Ignore"}
        </button>
      </div>
      {traffic.active && (
        <div className="traffic-chips">
          {cameras.slice(0, 3).map((camera) => (
            <span key={camera.id} className="badge traffic" title={`${camera.banner} · ${camera.videos.join(", ")}`}>
              🚦 {camera.label}
            </span>
          ))}
          {cameras.length > 3 && <span className="badge traffic">+{cameras.length - 3} cameras</span>}
          {traffic.dates.map((date) => (
            <span key={date} className="badge traffic">📅 {date}</span>
          ))}
          {traffic.time && (
            <span className="badge traffic" title={`from "${traffic.time.text}"`}>
              🕑 {traffic.time.from}–{traffic.time.to}
            </span>
          )}
          {traffic.race_stage != null && <span className="badge traffic">🚴 Stage {traffic.race_stage}</span>}
        </div>
      )}
      <div className="hint-text" data-testid="traffic-reason">{traffic.reason_en}</div>
      {traffic.warnings.map((warning) => (
        <div key={warning} className="hint-text warn">⚠ {warning}</div>
      ))}
      {mode !== traffic.mode && <div className="hint-text">Applies to the next <b>Search</b>.</div>}
    </div>
  );
}

export function QueryUnderstanding({
  parsed,
  traffic = null,
  trafficMode = "auto",
  onTrafficMode,
}: {
  parsed: ParsedQuery | null;
  traffic?: TrafficFilterInfo | null;
  trafficMode?: TrafficFilterMode;
  onTrafficMode?: (mode: TrafficFilterMode) => void;
}) {
  if (!parsed) return null;
  const showTraffic =
    traffic != null
    && onTrafficMode != null
    && (traffic.active || traffic.warnings.length > 0 || traffic.mode === "off" || trafficMode !== traffic.mode);
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
      {showTraffic && <TrafficFilter traffic={traffic!} mode={trafficMode} onMode={onTrafficMode!} />}
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
