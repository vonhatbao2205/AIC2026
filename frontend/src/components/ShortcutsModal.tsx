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
      { keys: ["←"], desc: "Previous frame in group" },
      { keys: ["→"], desc: "Next frame in group" },
      { keys: ["Tab"], desc: "Switch focus: results ⇄ detail" },
    ],
  },
  {
    title: "Video & timeline",
    rows: [
      { keys: ["v"], desc: "Show / hide video at the selected keyframe" },
      { keys: ["Space"], desc: "Play / pause the video (when focused)" },
      { keys: ["T"], desc: "Toggle the timeline" },
    ],
  },
  {
    title: "Submit",
    rows: [
      { keys: ["Enter"], desc: "Open submit guard (a result is selected)" },
      { keys: ["Enter"], desc: "Confirm submit (inside the guard)" },
      { keys: ["Enter"], desc: "TRAKE: assign snapped frame to active slot" },
    ],
  },
  {
    title: "General",
    rows: [
      { keys: ["Esc"], desc: "Close modal / cancel chip / close this help" },
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
