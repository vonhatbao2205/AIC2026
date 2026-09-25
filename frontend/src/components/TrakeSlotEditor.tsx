import { forwardRef, useEffect, useRef } from "react";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import type { PausedFrame } from "./PausedFramePanel";
import type { TrakeSlot } from "./TrakePanel";
import { VideoViewer, type VideoViewerHandle } from "./VideoViewer";

/** Which TRAKE event slot the editor is re-picking, and the video it plays. */
export interface SlotEditorState {
  slotIndex: number;
  video_id: string;
  video_url: string;
  fallback_url: string | null;
  /** Where the player opened: the slot's frame at the time `v` was pressed. It
   *  does not follow later replacements, or every Replace would seek the video. */
  start: number;
  fps: number;
}

interface Props {
  editor: SlotEditorState;
  /** The slot as it is now (it changes when the operator replaces its frame). */
  slot: TrakeSlot | null;
  pausedFrame: PausedFrame | null;
  onPaused: (frame: PausedFrame) => void;
  onReplace: () => void;
  onClose: () => void;
  /** A click in the editor gives it the scrub keys back. */
  onFocus: () => void;
}

/** Re-pick one TRAKE event's frame on its own video.
 *
 *  Opened with `v` after clicking a slot. The video opens parked on the slot's
 *  frame; `a`/`d`, the arrows and Space drive it like the main player, every stop
 *  is captured as the paused frame, and Enter (or Replace) swaps it into the slot.
 *  It works for any slot, including one whose video is not among the results. */
export const TrakeSlotEditor = forwardRef<VideoViewerHandle, Props>(function TrakeSlotEditor(props, ref) {
  const { editor, slot, pausedFrame } = props;
  const panelRef = useRef<HTMLElement>(null);
  const label = `E${editor.slotIndex + 1}`;
  const pausedHere = pausedFrame && pausedFrame.video_id === editor.video_id ? pausedFrame : null;

  useEffect(() => {
    // jsdom has no layout, hence the optional call.
    panelRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [editor.slotIndex, editor.video_id]);

  return (
    <section
      ref={panelRef}
      className="slot-editor"
      onPointerDown={props.onFocus}
      data-testid="trake-slot-editor"
    >
      <div className="slot-editor-head">
        <span className="event-dot" style={{ background: eventColor(editor.slotIndex) }} />
        <b>Editing {label}</b>
        <span className="mono">{editor.video_id}</span>
        {slot && (
          <span className="mono slot-editor-current">
            current f{slot.frame_idx} · {formatTime(slot.pts_time)}
          </span>
        )}
        <span className="slot-editor-keys">
          <span className="kbd">a</span>/<span className="kbd">d</span> ±1s · <span className="kbd">←</span>/
          <span className="kbd">→</span> ±5s · <span className="kbd">Space</span> play ·{" "}
          <span className="kbd">↵</span> replace
        </span>
        <button className="btn sm ghost" onClick={props.onClose}>Close</button>
      </div>
      <VideoViewer
        ref={ref}
        src={editor.video_url}
        fallbackSrc={editor.fallback_url}
        startTime={editor.start}
        onPaused={(rawTime, thumbnail) =>
          props.onPaused({
            video_id: editor.video_id,
            frame_idx: Math.round(rawTime * editor.fps),
            pts_time: rawTime,
            fps: editor.fps,
            thumbnail,
          })
        }
      />
      <div className="slot-editor-bar" data-testid="slot-editor-bar">
        {pausedHere ? (
          <>
            {pausedHere.thumbnail && <img src={pausedHere.thumbnail} alt={`Paused frame ${pausedHere.frame_idx}`} />}
            <span className="mono">
              Paused frame <b>{pausedHere.frame_idx}</b> · {formatTime(pausedHere.pts_time)} ·{" "}
              {pausedHere.pts_time.toFixed(3)}s
            </span>
            <button className="btn sm primary" onClick={props.onReplace} data-testid="slot-editor-replace">
              Replace {label} <span className="kbd">↵</span>
            </button>
          </>
        ) : (
          <span className="agent-note">Pause, or seek while paused, to capture a frame for {label}.</span>
        )}
      </div>
    </section>
  );
});
