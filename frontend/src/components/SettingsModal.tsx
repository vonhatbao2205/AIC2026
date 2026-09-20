import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ConfigImportResult, ConfigStatus } from "../api/types";
import { getDisplayName, isSupabaseConfigured, setDisplayName } from "../lib/supabase";

interface Props {
  open: boolean;
  onClose: () => void;
  /** Fired after a successful import so the console can re-read /api/health. */
  onApplied?: () => void;
}

/**
 * Configuration screen for the packaged app.
 *
 * The download-and-run build ships with no credentials: the operator imports the
 * `.env` their team shares. Everything here goes through the backend — the
 * browser never holds a key, and what comes back for secrets is masked.
 */
export function SettingsModal({ open, onClose, onApplied }: Props) {
  const [status, setStatus] = useState<ConfigStatus | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [replace, setReplace] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ConfigImportResult | null>(null);
  const [dragging, setDragging] = useState(false);
  const [name, setName] = useState(getDisplayName);
  const [nameSaved, setNameSaved] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.config());
    } catch {
      setError("Could not read backend configuration.");
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void refresh();
    setName(getDisplayName());
    setNameSaved(false);
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  async function doImport() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.importConfig(file, replace);
      setResult(res);
      setStatus(res.status);
      setFile(null);
      onApplied?.();
    } catch (e) {
      setError(
        e instanceof ApiError && typeof e.detail === "string"
          ? e.detail
          : "Import failed — check the .env file.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function doReload() {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.reloadConfig());
      onApplied?.();
    } catch {
      setError("Reload failed.");
    } finally {
      setBusy(false);
    }
  }

  function saveName() {
    setDisplayName(name);
    setName(getDisplayName());
    setNameSaved(true);
  }

  if (!open) return null;

  const missing = status?.missing_required ?? [];
  const needsSetup = status ? !status.configured : false;

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="settings-modal">
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: 720 }}>
        <div className="modal-head">
          <h2>Configuration</h2>
          <button className="btn sm ghost" onClick={onClose}>
            esc
          </button>
        </div>

        <div className="modal-body">
          {needsSetup && (
            <div className="cfg-banner bad" data-testid="settings-needs-setup">
              <strong>Not configured.</strong> Import your team's <code>.env</code> file to connect
              Elastic, Milvus, encoders and media. Missing:{" "}
              <span className="mono">{missing.join(", ")}</span>
            </div>
          )}
          {status?.mock_mode && (
            <div className="cfg-banner warn">
              Running in <strong>mock mode</strong> — results are fixed fixtures, not live
              data. Set <span className="mono">AIC26_MOCK_MODE=false</span> in .env to use live services.
            </div>
          )}

          <div className="cfg-group" data-testid="settings-identity">
            <div className="cfg-group-head">
              <span className="cfg-group-name">Your name</span>
              <span className="cfg-group-sum">
                Shown beside each row you add in the Submission tab.
              </span>
            </div>
            <div className="cfg-name-row">
              <input
                className="cfg-name-input"
                type="text"
                value={name}
                placeholder="e.g. Bao"
                maxLength={40}
                data-testid="settings-name-input"
                onChange={(e) => {
                  setName(e.target.value);
                  setNameSaved(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") saveName();
                }}
              />
              <button
                className="btn sm primary"
                onClick={saveName}
                data-testid="settings-name-save"
              >
                Save name
              </button>
            </div>
            <div className="cfg-name-hint">
              {nameSaved ? (
                <span className="cfg-name-ok" data-testid="settings-name-saved">
                  Saved{name.trim() ? "" : " — leave blank to display “unknown” on your rows"}.
                  Takes effect immediately; no reload required.
                </span>
              ) : isSupabaseConfigured() ? (
                "Saved on this device only. Set a name once on each team device."
              ) : (
                "Supabase sync is disabled; this name is only used for local submissions."
              )}
            </div>
          </div>

          <div
            className={`cfg-drop ${dragging ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const dropped = e.dataTransfer.files?.[0];
              if (dropped) setFile(dropped);
            }}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              hidden
              data-testid="settings-file-input"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? (
              <>
                <div className="cfg-drop-main mono">{file.name}</div>
                <div className="cfg-drop-sub">{(file.size / 1024).toFixed(1)} KB — ready to import</div>
              </>
            ) : (
              <>
                <div className="cfg-drop-main">Drop a .env file here</div>
                <div className="cfg-drop-sub">or click to choose a file</div>
              </>
            )}
          </div>

          <div className="cfg-actions">
            <label className="cfg-check" title="Remove existing values and use only those in the file">
              <input
                type="checkbox"
                checked={replace}
                onChange={(e) => setReplace(e.target.checked)}
              />
              Replace entire configuration
            </label>
            <div className="spacer" />
            <a className="btn sm ghost" href={api.configTemplateUrl()} download=".env">
              Download .env template
            </a>
            <button className="btn sm ghost" onClick={doReload} disabled={busy}>
              Reload from disk
            </button>
            <button
              className="btn sm primary"
              onClick={doImport}
              disabled={!file || busy}
              data-testid="settings-import"
            >
              {busy ? "Applying…" : "Import & apply"}
            </button>
          </div>

          {error && <div className="cfg-banner bad">{error}</div>}
          {result && (
            <div className="cfg-banner ok" data-testid="settings-result">
              Applied <strong>{result.applied.length}</strong> variables
              {result.replaced ? " (full replacement)" : ""}.
              {result.ignored_blank.length > 0 &&
                ` Skipped ${result.ignored_blank.length} blank values (existing values kept).`}
              {result.unknown.length > 0 && ` Unknown configuration keys: ${result.unknown.join(", ")}.`}
              {result.rejected.length > 0 && ` Rejected: ${result.rejected.join(", ")}.`}
            </div>
          )}

          {status && (
            <>
              <div className="cfg-path">
                Write to <span className="mono">{status.env_path}</span>
                {status.env_exists ? "" : " (does not exist yet)"}
              </div>
              {status.groups.map((group) => (
                <div className="cfg-group" key={group.name}>
                  <div className="cfg-group-head">
                    <span className="cfg-group-name">{group.name}</span>
                    <span className="cfg-group-sum">{group.summary}</span>
                  </div>
                  <table className="cfg-table">
                    <tbody>
                      {group.keys.map((k) => (
                        <tr key={k.key} className={k.required && !k.set ? "missing" : ""}>
                          <td className="cfg-key mono">
                            <span
                              className={`health-dot ${k.set ? "ok" : k.required ? "bad" : "warn"}`}
                            />
                            {k.key}
                            {k.required && <span className="cfg-req" title="required">*</span>}
                          </td>
                          <td className="cfg-label">{k.label}</td>
                          <td className="cfg-value mono">
                            {k.set ? k.preview : <span className="cfg-unset">—</span>}
                            {k.from_process_env && (
                              <span
                                className="cfg-envtag"
                                title="Set by a container environment variable; importing a file cannot override it"
                              >
                                env
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
