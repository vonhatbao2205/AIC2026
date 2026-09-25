interface Props {
  open: boolean;
  onClose: () => void;
}

type Row = { keys: string[]; desc: string };
type Group = { title: string; rows: Row[] };

const GROUPS: Group[] = [
  {
    title: "Search & query",
    rows: [
      { keys: ["/"], desc: "Focus the query box" },
      { keys: ["Enter"], desc: "Search (while typing in the query)" },
      { keys: ["Ctrl", "M"], desc: "Voice input (if supported)" },
    ],
  },
  {
    title: "Navigation",
    rows: [
      { keys: ["↑"], desc: "Previous video group" },
      { keys: ["↓"], desc: "Next video group" },
      { keys: ["←"], desc: "Previous frame in group (when no video is open)" },
      { keys: ["→"], desc: "Next frame in group (when no video is open)" },
      { keys: ["Tab"], desc: "Switch focus: results ⇄ detail" },
    ],
  },
  {
    title: "Video & timeline",
    rows: [
      { keys: ["v"], desc: "Show / hide video at the selected keyframe" },
      { keys: ["Space"], desc: "Play / pause (the video opens paused on the frame)" },
      { keys: ["T"], desc: "Toggle the timeline" },
      { keys: ["K"], desc: "Browse neighbouring keyframes of the selected frame" },
      { keys: ["←", "→"], desc: "Page the neighbour strip (video follows) when it is open" },
      { keys: ["←", "→"], desc: "Rewind / forward the open video by 5s (strip closed)" },
      { keys: ["a", "d"], desc: "Rewind / forward the open video by 1s" },
    ],
  },
  {
    title: "Submit",
    rows: [
      { keys: ["Enter"], desc: "Submit the selected result keyframe (shown in Detail)" },
      { keys: ["Shift", "Enter"], desc: "Submit the captured paused raw frame" },
      { keys: ["Enter"], desc: "Confirm submit (inside the guard)" },
      { keys: ["Enter"], desc: "TRAKE: assign paused frame to active slot" },
    ],
  },
  {
    title: "V-KIS sketch (while the canvas has focus)",
    rows: [
      { keys: ["B", "E"], desc: "Brush / eraser" },
      { keys: ["L", "R", "O"], desc: "Line / rectangle / ellipse (hold Shift to snap)" },
      { keys: ["F", "G"], desc: "Freeform fill / bucket fill" },
      { keys: ["I"], desc: "Eyedropper (Alt+click works with every tool)" },
      { keys: ["[", "]"], desc: "Smaller / larger brush" },
      { keys: ["1", "…", "0"], desc: "Opacity 10% … 100%" },
      { keys: ["Ctrl", "Z"], desc: "Undo (Ctrl+Shift+Z or Ctrl+Y: redo)" },
      { keys: ["Ctrl", "Enter"], desc: "Search with the sketch" },
    ],
  },
  {
    title: "General",
    rows: [
      { keys: ["`"], desc: "Open / close the local Sticky Note" },
      { keys: ["Esc"], desc: "Close modal / clear paused frame / close this help" },
      { keys: ["Ctrl", "/"], desc: "Toggle this shortcuts help" },
    ],
  },
];

export function ShortcutsModal({ open, onClose }: Props) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="shortcuts-modal">
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: 620 }}>
        <div className="modal-head">
          <h2>Keyboard shortcuts</h2>
          <button className="btn sm ghost" onClick={onClose}>esc</button>
        </div>
        <div className="modal-body">
          <div className="keymap-grid">
            {GROUPS.map((g) => (
              <div key={g.title} className="keymap-group">
                <div className="keymap-title">{g.title}</div>
                {g.rows.map((r, i) => (
                  <div className="keymap-row" key={i}>
                    <span className="keymap-keys">
                      {r.keys.map((k, j) => (
                        <span key={j}>
                          <kbd className="kbd">{k}</kbd>
                          {j < r.keys.length - 1 && <span className="keymap-plus">+</span>}
                        </span>
                      ))}
                    </span>
                    <span className="keymap-desc">{r.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
