import type { Channel, ParsedQuery, RetrievalDatabase } from "../api/types";
import { CHANNELS, CHANNEL_TOGGLE_LABEL } from "../lib/constants";

interface Props {
  parsed: ParsedQuery | null;
  overrides: { force_channels: string[]; disable_channels: string[] };
  onToggle: (channel: Channel, enabled: boolean) => void;
  retrievalDatabase: RetrievalDatabase;
}

/** The `image_pe` switch is the master switch for BOTH image indices: the backend
 *  reads `channels.image_pe.enabled` for the Qwen channel too. Say so on hover,
 *  because switching it off during a Qwen search silently kills every keyframe. */
function toggleHint(channel: Channel, disabled: boolean): string | undefined {
  if (disabled) return "Object detection is not available for InfoShot++";
  if (channel === "image_pe") {
    return "Master visual-channel switch — disabling it turns off both PE Core and Qwen3-VL. Choose indices under Image embedding.";
  }
  return undefined;
}

export function ChannelControls({ parsed, overrides, onToggle, retrievalDatabase }: Props) {
  // InfoShot++ answers OCR / speech / audio from its own v2 indices since the
  // metadata migration. Only the OD-backed channels stay unavailable there.
  const unsupported = new Set<Channel>(["object_layout", "canvas_image"]);
  function isEnabled(c: Channel): boolean {
    if (overrides.disable_channels.includes(c)) return false;
    if (overrides.force_channels.includes(c)) return true;
    return parsed?.channels?.[c]?.enabled ?? c === "image_pe";
  }

  return (
    <div className="panel">
      <h3>Retrieval channels</h3>
      {CHANNELS.filter((channel) => channel !== "tara").map((c) => {
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
            title={toggleHint(c, disabled)}
          >
            <span className="lbl">
              <span className={`badge ${c}`}>{CHANNEL_TOGGLE_LABEL[c]}</span>
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
          ? "VISUAL controls keyframe retrieval for both PE Core and Qwen3-VL. Choose PE/Qwen/TARA under Visual models. InfoShot++ also supports similar-image search and OCR/speech/audio on index v2. OCR excludes L26; V-KIS canvas is unavailable without object detection."
          : "Confidence/stoplist demotion (speech low/intro, audio stoplist & generic captions) is applied automatically by the backend."}
      </div>
    </div>
  );
}
