import { useEffect, useRef, useState } from "react";
import { type Hsv, clampByte, contrastInk, hexToHsv, hexToRgb, hsvToHex, normalizeHex, rgbToHex } from "../lib/color";

/** Colours a scene sketch keeps reaching for: the basics, then sky, water,
 *  vegetation, ground, road, brick and skin tones. */
export const PRESET_SWATCHES = [
  "#000000", "#404040", "#808080", "#c0c0c0", "#ffffff", "#e02020", "#ff8c1a", "#ffd83a",
  "#2e9e44", "#1e50dc", "#8a3ab9", "#f28cb8", "#87c3ec", "#3d85c6", "#e8eef4", "#2f6f8f",
  "#6aa84f", "#274e13", "#8b5a2b", "#e0c48a", "#555b63", "#a4452c", "#f1c9a5", "#a86b4c",
];

interface EyeDropperResult { sRGBHex: string }
interface EyeDropperApi { open: () => Promise<EyeDropperResult> }
type EyeDropperCtor = new () => EyeDropperApi;

function eyeDropper(): EyeDropperCtor | null {
  const ctor = (window as unknown as { EyeDropper?: EyeDropperCtor }).EyeDropper;
  return typeof ctor === "function" ? ctor : null;
}

interface Props {
  title: string;
  value: string;
  /** Applied live while the picker is open, so the brush previews the colour. */
  onChange: (hex: string) => void;
  /** `accepted` is false for Cancel/Esc, after `onChange` restored the colour. */
  onClose: (accepted: boolean) => void;
  recent: string[];
  custom: string[];
  onAddCustom: (hex: string) => void;
  onRemoveCustom: (hex: string) => void;
  placement?: "below" | "above";
}

/** Photoshop-style picker: saturation × brightness square, vertical hue strip,
 *  new/current comparison and exact H/S/B, R/G/B and hex entry. The HSV state is
 *  kept here rather than re-derived from the hex, so dragging through black or
 *  grey does not throw the hue away. */
export function ColorPicker({
  title, value, onChange, onClose, recent, custom, onAddCustom, onRemoveCustom, placement = "below",
}: Props) {
  const original = useRef(value);
  const [hsv, setHsvState] = useState<Hsv>(() => hexToHsv(value));
  const [hexDraft, setHexDraft] = useState<string | null>(null);
  const hex = hsvToHex(hsv);
  const rgb = hexToRgb(hex);

  const setHsv = (next: Hsv) => {
    const clamped = {
      h: Math.min(360, Math.max(0, next.h)),
      s: Math.min(100, Math.max(0, next.s)),
      v: Math.min(100, Math.max(0, next.v)),
    };
    setHsvState(clamped);
    onChange(hsvToHex(clamped));
  };
  const setHex = (next: string) => {
    const normalized = normalizeHex(next);
    if (!normalized) return;
    // Keep the hue for greys, which have none of their own.
    const parsed = hexToHsv(normalized);
    setHsvState(parsed.s === 0 || parsed.v === 0 ? { ...parsed, h: hsv.h } : parsed);
    onChange(normalized);
  };

  const cancel = () => {
    onChange(original.current);
    onClose(false);
  };
  const accept = () => onClose(true);

  // Outside clicks close the picker keeping the colour, like clicking OK. The
  // element it is rendered in (its anchor, holding the button that opened it)
  // counts as inside, so that button toggles it instead of closing it and
  // reopening it in the same click.
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function onDown(event: PointerEvent) {
      const boundary = rootRef.current?.parentElement ?? rootRef.current;
      if (boundary && event.target instanceof Node && !boundary.contains(event.target)) onClose(true);
    }
    document.addEventListener("pointerdown", onDown, true);
    return () => document.removeEventListener("pointerdown", onDown, true);
  }, [onClose]);

  function dragger(apply: (fx: number, fy: number) => void) {
    return {
      onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
        event.preventDefault();
        const element = event.currentTarget;
        element.setPointerCapture?.(event.pointerId);
        element.focus({ preventScroll: true });
        const move = (clientX: number, clientY: number) => {
          const rect = element.getBoundingClientRect();
          if (!rect.width || !rect.height) return;
          apply(
            Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)),
            Math.min(1, Math.max(0, (clientY - rect.top) / rect.height)),
          );
        };
        move(event.clientX, event.clientY);
        const onMove = (e: PointerEvent) => move(e.clientX, e.clientY);
        const onUp = () => {
          element.removeEventListener("pointermove", onMove);
          element.removeEventListener("pointerup", onUp);
          element.removeEventListener("pointercancel", onUp);
        };
        element.addEventListener("pointermove", onMove);
        element.addEventListener("pointerup", onUp);
        element.addEventListener("pointercancel", onUp);
      },
    };
  }

  const svKeys = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 10 : 1;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step],
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    event.stopPropagation();
    setHsv({ ...hsv, s: hsv.s + move[0], v: hsv.v + move[1] });
  };
  const hueKeys = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 10 : 1;
    const delta = event.key === "ArrowUp" ? step : event.key === "ArrowDown" ? -step : 0;
    if (!delta) return;
    event.preventDefault();
    event.stopPropagation();
    setHsv({ ...hsv, h: (hsv.h + delta + 360) % 360 });
  };

  const number = (
    label: string,
    current: number,
    max: number,
    apply: (n: number) => void,
    suffix = "",
  ) => (
    <label className="cp-field">
      <span>{label}</span>
      <input
        type="number"
        min={0}
        max={max}
        value={Math.round(current)}
        onChange={(event) => {
          const n = Number(event.target.value);
          if (event.target.value !== "" && Number.isFinite(n)) apply(Math.min(max, Math.max(0, n)));
        }}
        data-testid={`cp-${label.toLowerCase()}${suffix ? "-" + suffix : ""}`}
      />
      {suffix === "pct" ? <em>%</em> : suffix === "deg" ? <em>°</em> : <em />}
    </label>
  );
  const setRgb = (channel: "r" | "g" | "b", n: number) => setHex(rgbToHex({ ...rgb, [channel]: clampByte(n) }));

  const Dropper = eyeDropper();
  const swatch = (color: string, key: string, extra?: Partial<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button
      key={key}
      type="button"
      className={`swatch${color === hex ? " on" : ""}`}
      style={{ background: color }}
      title={color}
      onClick={() => setHex(color)}
      {...extra}
    />
  );

  return (
    <div
      ref={rootRef}
      className={`color-picker ${placement}`}
      role="dialog"
      aria-label={title}
      data-testid="color-picker"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          cancel();
        } else if (event.key === "Enter" && !(event.target instanceof HTMLInputElement)) {
          event.preventDefault();
          event.stopPropagation();
          accept();
        }
      }}
    >
      <div className="cp-title">
        <strong>{title}</strong>
        <button type="button" className="cp-x" onClick={cancel} title="Cancel (Esc)">×</button>
      </div>
      <div className="cp-body">
        <div
          className="cp-sv"
          style={{ background: `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hsv.h} 100% 50%))` }}
          tabIndex={0}
          role="slider"
          aria-label="Saturation and brightness"
          aria-valuetext={`S ${Math.round(hsv.s)}%, B ${Math.round(hsv.v)}%`}
          onKeyDown={svKeys}
          data-testid="cp-sv"
          {...dragger((fx, fy) => setHsv({ ...hsv, s: fx * 100, v: (1 - fy) * 100 }))}
        >
          <span
            className="cp-sv-knob"
            style={{ left: `${hsv.s}%`, top: `${100 - hsv.v}%`, borderColor: contrastInk(hex) }}
          />
        </div>
        <div
          className="cp-hue"
          tabIndex={0}
          role="slider"
          aria-label="Hue"
          aria-valuemin={0}
          aria-valuemax={360}
          aria-valuenow={Math.round(hsv.h)}
          onKeyDown={hueKeys}
          data-testid="cp-hue"
          {...dragger((_, fy) => setHsv({ ...hsv, h: (1 - fy) * 360 }))}
        >
          <span className="cp-hue-knob" style={{ top: `${100 - (hsv.h / 360) * 100}%` }} />
        </div>
        <div className="cp-side">
          <div className="cp-compare" title="new / current — click current to go back to it">
            <span style={{ background: hex, color: contrastInk(hex) }} data-testid="cp-new">new</span>
            <button
              type="button"
              style={{ background: original.current, color: contrastInk(original.current) }}
              onClick={() => setHex(original.current)}
              data-testid="cp-current"
            >
              current
            </button>
          </div>
          <div className="cp-fields">
            {number("H", hsv.h, 360, (h) => setHsv({ ...hsv, h }), "deg")}
            {number("R", rgb.r, 255, (n) => setRgb("r", n))}
            {number("S", hsv.s, 100, (s) => setHsv({ ...hsv, s }), "pct")}
            {number("G", rgb.g, 255, (n) => setRgb("g", n))}
            {number("B", hsv.v, 100, (v) => setHsv({ ...hsv, v }), "pct")}
            {number("B", rgb.b, 255, (n) => setRgb("b", n), "blue")}
          </div>
          <label className="cp-field cp-hex">
            <span>#</span>
            <input
              value={hexDraft ?? hex.slice(1)}
              maxLength={7}
              spellCheck={false}
              onFocus={(event) => { setHexDraft(hex.slice(1)); event.target.select(); }}
              onChange={(event) => {
                setHexDraft(event.target.value);
                if (/^#?[0-9a-fA-F]{6}$/.test(event.target.value.trim())) setHex(event.target.value);
              }}
              onBlur={() => {
                if (hexDraft !== null) setHex(hexDraft);
                setHexDraft(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  event.stopPropagation();
                  if (hexDraft !== null) setHex(hexDraft);
                  setHexDraft(null);
                  accept();
                }
              }}
              data-testid="cp-hex"
            />
          </label>
        </div>
      </div>
      <div className="cp-swatches" data-testid="cp-presets">
        {PRESET_SWATCHES.map((color) => swatch(color, `p${color}`))}
      </div>
      {custom.length > 0 && (
        <div className="cp-row">
          <span className="cp-label">Saved</span>
          <div className="cp-swatches">
            {custom.map((color) =>
              swatch(color, `c${color}`, {
                title: `${color} — Shift+click to remove`,
                onClick: (event) => (event.shiftKey ? onRemoveCustom(color) : setHex(color)),
              }),
            )}
          </div>
        </div>
      )}
      {recent.length > 0 && (
        <div className="cp-row">
          <span className="cp-label">Recent</span>
          <div className="cp-swatches">{recent.map((color) => swatch(color, `r${color}`))}</div>
        </div>
      )}
      <div className="cp-actions">
        <button type="button" className="btn sm" onClick={() => onAddCustom(hex)} data-testid="cp-add">
          + Add to swatches
        </button>
        {Dropper && (
          <button
            type="button"
            className="btn sm"
            title="Pick a colour from anywhere on screen (e.g. the query clip)"
            onClick={() => {
              new Dropper().open().then((result) => setHex(result.sRGBHex)).catch(() => {});
            }}
          >
            ◉ Screen
          </button>
        )}
        <span style={{ flex: 1 }} />
        <button type="button" className="btn sm" onClick={cancel}>Cancel</button>
        <button type="button" className="btn sm primary" onClick={accept} data-testid="cp-ok">OK</button>
      </div>
    </div>
  );
}

