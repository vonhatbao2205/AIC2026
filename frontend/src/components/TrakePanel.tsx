import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import { hasPeakDrag, readPeakDrag, type TrakePeakDrag } from "../lib/trakeDrag";

// A TRAKE slot now holds the RAW paused frame (frame_idx computed from pts*fps),
// not a snapped BTC keyframe — TRAKE may need frames outside the BTC keyframe set.
export interface TrakeSlot {
  video_id: string;
  frame_idx: number;
  pts_time: number;
  /** Direct keyframe URL, or an ephemeral data URL captured from an exact raw frame. */
  thumbnail?: string | null;
  submit_keyframe_id?: string; // extracted keyframe provenance, absent for raw frames
  /** Whether a restored thumbnail came from its keyframe image or was freshly
   *  extracted from the exact raw-frame timestamp. */
  thumbnail_kind?: "exact" | "exact-raw";
}

interface Props {
  slots: (TrakeSlot | null)[];
  activeSlot: number;
  hasPausedFrame: boolean;
  violations: number[];
  onSetActive: (i: number) => void;
  onAssignChip: (slotIdx: number) => void;
  /** A frame dragged straight out of a TRAKE result card (event representative
   *  or heat peak). It goes into the slot as-is — no video load, no seek, no
   *  pause, which is the whole point of showing the frames in the first place. */
  onAssignPeak: (slotIdx: number, payload: TrakePeakDrag) => void;
  onClearSlot: (i: number) => void;
  onMoveSlot: (from: number, to: number) => void;
  onAddEvent: () => void;
  onRemoveEvent: () => void;
}

function SlotThumb({ slot }: { slot: TrakeSlot }) {
  if (slot.thumbnail) {
    return (
      <div className="trake-slot-thumb">
        <img src={slot.thumbnail} alt={`frame ${slot.frame_idx}`} />
        {slot.thumbnail_kind === "exact-raw" && (
          <span title="Ảnh được trích lại từ video tại đúng thời điểm raw frame">
            raw exact
          </span>
        )}
      </div>
    );
  }
  return (
    <div style={{ height: 54, borderRadius: 5, background: "var(--bg-3)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--fg-faint)", fontSize: 9 }}>
      no preview
    </div>
  );
}

export function TrakePanel(props: Props) {
  const { slots, activeSlot, violations } = props;

  function onDrop(e: React.DragEvent, slotIdx: number) {
    e.preventDefault();
    const fromSlot = e.dataTransfer.getData("text/x-slot");
    if (fromSlot !== "") {
      const from = Number(fromSlot);
      if (from !== slotIdx) props.onMoveSlot(from, slotIdx); // reorder events
      return;
    }
    const peak = readPeakDrag(e);
    if (peak) {
      props.onAssignPeak(slotIdx, peak);
      return;
    }
    if (e.dataTransfer.getData("text/x-paused-frame") && props.hasPausedFrame) {
      props.onAssignChip(slotIdx);
    }
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
            onDragOver={(e) => {
              e.preventDefault();
              if (hasPeakDrag(e)) e.dataTransfer.dropEffect = "copy";
            }}
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
                empty — kéo frame từ kết quả, hoặc pause video & Enter
              </div>
            )}
          </div>
        ))}
      </div>

    </div>
  );
}
