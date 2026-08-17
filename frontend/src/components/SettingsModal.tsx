import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ConfigImportResult, ConfigStatus } from "../api/types";

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
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.config());
    } catch {
      setError("Không đọc được cấu hình từ backend.");
    }
  }, []);

  useEffect(() => {
    if (open) void refresh();
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
          : "Import thất bại — kiểm tra lại file .env.",
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
      setError("Reload thất bại.");
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  const missing = status?.missing_required ?? [];
  const needsSetup = status ? !status.configured : false;

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="settings-modal">
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: 720 }}>
        <div className="modal-head">
          <h2>Cấu hình · Configuration</h2>
          <button className="btn sm ghost" onClick={onClose}>
            esc
          </button>
        </div>

        <div className="modal-body">
          {needsSetup && (
            <div className="cfg-banner bad" data-testid="settings-needs-setup">
              <strong>Chưa cấu hình.</strong> Import file <code>.env</code> của nhóm để kết nối
              Elastic, Milvus, encoder và media. Thiếu:{" "}
              <span className="mono">{missing.join(", ")}</span>
            </div>
          )}
          {status?.mock_mode && (
            <div className="cfg-banner warn">
              Đang chạy <strong>mock mode</strong> — kết quả là fixture cố định, không phải dữ liệu
              thật. Đặt <span className="mono">AIC26_MOCK_MODE=false</span> trong .env để dùng thật.
            </div>
          )}

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
                <div className="cfg-drop-sub">{(file.size / 1024).toFixed(1)} KB — sẵn sàng import</div>
              </>
            ) : (
              <>
                <div className="cfg-drop-main">Kéo thả file .env vào đây</div>
                <div className="cfg-drop-sub">hoặc bấm để chọn file</div>
              </>
            )}
          </div>

          <div className="cfg-actions">
            <label className="cfg-check" title="Xoá mọi giá trị hiện có, chỉ giữ những gì trong file">
              <input
                type="checkbox"
                checked={replace}
                onChange={(e) => setReplace(e.target.checked)}
              />
              Thay thế toàn bộ cấu hình
            </label>
            <div className="spacer" />
            <a className="btn sm ghost" href={api.configTemplateUrl()} download=".env">
              Tải .env mẫu
            </a>
            <button className="btn sm ghost" onClick={doReload} disabled={busy}>
              Đọc lại từ đĩa
            </button>
            <button
              className="btn sm primary"
              onClick={doImport}
              disabled={!file || busy}
              data-testid="settings-import"
            >
              {busy ? "Đang áp dụng…" : "Import & áp dụng"}
            </button>
          </div>

          {error && <div className="cfg-banner bad">{error}</div>}
          {result && (
            <div className="cfg-banner ok" data-testid="settings-result">
              Đã áp dụng <strong>{result.applied.length}</strong> biến
              {result.replaced ? " (thay thế toàn bộ)" : ""}.
              {result.ignored_blank.length > 0 &&
                ` Bỏ qua ${result.ignored_blank.length} dòng trống (giá trị cũ được giữ).`}
              {result.unknown.length > 0 && ` Không nằm trong schema: ${result.unknown.join(", ")}.`}
              {result.rejected.length > 0 && ` Từ chối: ${result.rejected.join(", ")}.`}
            </div>
          )}

          {status && (
            <>
              <div className="cfg-path">
                Ghi vào <span className="mono">{status.env_path}</span>
                {status.env_exists ? "" : " (chưa tồn tại)"}
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
                            {k.required && <span className="cfg-req" title="bắt buộc">*</span>}
                          </td>
                          <td className="cfg-label">{k.label}</td>
                          <td className="cfg-value mono">
                            {k.set ? k.preview : <span className="cfg-unset">—</span>}
                            {k.from_process_env && (
                              <span
                                className="cfg-envtag"
                                title="Đến từ biến môi trường của container — import file sẽ không đổi được giá trị này"
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
