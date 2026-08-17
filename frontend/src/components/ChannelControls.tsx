import type { Channel, ParsedQuery, RetrievalDatabase } from "../api/types";
import { CHANNELS, CHANNEL_LABEL } from "../lib/constants";

interface Props {
  parsed: ParsedQuery | null;
  overrides: { force_channels: string[]; disable_channels: string[] };
  onToggle: (channel: Channel, enabled: boolean) => void;
  retrievalDatabase: RetrievalDatabase;
}

export function ChannelControls({ parsed, overrides, onToggle, retrievalDatabase }: Props) {
  const unsupported = new Set<Channel>(["ocr", "speech", "audio", "object_layout", "canvas_image"]);
  function isEnabled(c: Channel): boolean {
    if (overrides.disable_channels.includes(c)) return false;
    if (overrides.force_channels.includes(c)) return true;
    return parsed?.channels?.[c]?.enabled ?? c === "image_pe";
  }

  return (
    <div className="panel">
      <h3>Retrieval channels</h3>
      {CHANNELS.map((c) => {
        const disabled = retrievalDatabase === "infoshotpp" && unsupported.has(c);
        const on = disabled ? false : isEnabled(c);
        const cfg = parsed?.channels?.[c];
        const auto = parsed?.channels?.[c]?.enabled;
        return (
          <div
            key={c}
            className={`toggle ${on ? "on" : "off"} ${disabled ? "disabled" : ""}`}
            onClick={() => { if (!disabled) onToggle(c, !on); }}
            role="switch"
            aria-checked={on}
            data-testid={`channel-${c}`}
            aria-disabled={disabled}
            title={disabled ? "Chưa có dữ liệu cho InfoShot++" : undefined}
          >
            <span className="lbl">
              <span className={`badge ${c}`}>{CHANNEL_LABEL[c]}</span>
              {auto && <span style={{ fontSize: 9, color: "var(--fg-faint)" }}>auto</span>}
            </span>
            <span className="row" style={{ gap: 6 }}>
              {cfg?.weight ? <span className="weight">w{cfg.weight.toFixed(1)}</span> : null}
              <span className="sw" />
            </span>
          </div>
        );
      })}
      <div className="hint-text">
        {retrievalDatabase === "infoshotpp"
          ? "InfoShot++ chỉ chạy PE image và similar-image trên embedding mới."
          : "Confidence/stoplist demotion (speech low/intro, audio stoplist & generic captions) is applied automatically by the backend."}
      </div>
    </div>
  );
}
