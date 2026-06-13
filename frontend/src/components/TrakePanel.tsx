import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";

// A TRAKE slot now holds the RAW paused frame (frame_idx computed from pts*fps),
// not a snapped BTC keyframe — TRAKE may need frames outside the BTC keyframe set.
export interface TrakeSlot {
  video_id: string;
  frame_idx: number;
  pts_time: number;
  thumbnail?: string | null; // data URL captured from the exact paused frame
  submit_keyframe_id?: string; // nearest BTC keyframe, display only
}

export interface PendingChip {
  video_id: string;
  frame_idx: number;
  pts_time: number;
  thumbnail?: string | null;
}

interface Props {
  slots: (TrakeSlot | null)[];
  activeSlot: number;
  pendingChip: PendingChip | null;
  violations: number[];
  onSetActive: (i: number) => void;
  onAssignChip: (slotIdx: number) => void;
  onClearSlot: (i: number) => void;
  onMoveSlot: (from: number, to: number) => void;
  onAddEvent: () => void;
  onRemoveEvent: () => void;
}

function SlotThumb({ slot }: { slot: TrakeSlot }) {
  if (slot.thumbnail) return <img src={slot.thumbnail} alt={`frame ${slot.frame_idx}`} />;
  return (
    <div style={{ height: 54, borderRadius: 5, background: "var(--bg-3)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-faint)", fontSize: 9 }}>
      no preview
    </div>
  );
}

export function TrakePanel(props: Props) {
  const { slots, activeSlot, pendingChip, violations } = props;

  function onDrop(e: React.DragEvent, slotIdx: number) {
    e.preventDefault();
    const fromSlot = e.dataTransfer.getData("text/x-slot");
    if (fromSlot !== "") {
      const from = Number(fromSlot);
      if (from !== slotIdx) props.onMoveSlot(from, slotIdx); // reorder events
      return;
    }
    if (e.dataTransfer.getData("text/x-trake-chip")) props.onAssignChip(slotIdx);
  }

  return (
    <div className="panel" data-testid="trake-panel">
      <h3>TRAKE events</h3>
      <div className="row" style={{ marginBottom: 8 }}>
        <button className="btn sm" onClick={props.onAddEvent}>+ event</button>
        <button className="btn sm" onClick={props.onRemoveEvent} disabled={slots.length <= 1}>− event</button>
        {violations.length > 0 && (
          <span className="badge warn" title="Events must increase in frame index / time">
            ⚠ order: E{violations.join(", E")}
          </span>
        )}
      </div>

      <div className="trake-slots">
        {slots.map((slot, i) => (
          <div
            key={i}
            className={`slot ${i === activeSlot ? "active" : ""} ${slot ? "filled" : ""}`}
            data-testid={`trake-slot-${i}`}
            onClick={() => props.onSetActive(i)}
            draggable={!!slot}
            onDragStart={(e) => { if (slot) e.dataTransfer.setData("text/x-slot", String(i)); }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => onDrop(e, i)}
            style={{ ...(i === activeSlot ? { boxShadow: `0 0 0 2px ${eventColor(i)}` } : {}), cursor: slot ? "grab" : "pointer" }}
          >
            <div className="slot-head">
              <span className="row" style={{ gap: 4 }}>
                <span className="event-dot" style={{ background: eventColor(i) }} /> E{i + 1}
              </span>
              {slot && (
                <button className="btn sm ghost" style={{ padding: "0 4px" }} onClick={(e) => { e.stopPropagation(); props.onClearSlot(i); }}>✕</button>
              )}
            </div>
            {slot ? (
              <>
                <SlotThumb slot={slot} />
                <div className="slot-id">f{slot.frame_idx} · {formatTime(slot.pts_time)}</div>
              </>
            ) : (
              <div style={{ fontSize: 10, color: "var(--fg-faint)", textAlign: "center", padding: "16px 0" }}>
                empty — pause video & press Enter, or drag chip
              </div>
            )}
          </div>
        ))}
      </div>

      {pendingChip && (
        <div
          className="frame-chip"
          data-testid="frame-chip"
          draggable
          onDragStart={(e) => e.dataTransfer.setData("text/x-trake-chip", String(pendingChip.frame_idx))}
          title="Drag into a slot, or press Enter to assign to the active slot"
        >
          <div className="row">
            {pendingChip.thumbnail && <img src={pendingChip.thumbnail} style={{ width: 64, height: 36, objectFit: "cover", borderRadius: 4 }} alt="" />}
            <div>
              <div className="mono" style={{ fontSize: 11 }}>
                {pendingChip.video_id} · frame <b>{pendingChip.frame_idx}</b>
              </div>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--fg-faint)" }}>
                paused {pendingChip.pts_time.toFixed(3)}s → frame_idx = round(pts × fps)
              </div>
            </div>
          </div>
          <div style={{ fontSize: 9.5, color: "var(--fg-faint)", marginTop: 4 }}>
            <span className="kbd">↵</span> assign to E{activeSlot + 1}
          </div>
        </div>
      )}
    </div>
  );
}
