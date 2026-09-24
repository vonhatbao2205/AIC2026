import type { FrameResult, QaAnalysisResponse } from "../api/types";
import { formatTime } from "../lib/media";
import { qaVisionLabel } from "../lib/qa";

interface Props {
  analysis: QaAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  available: boolean | null;
  candidateCount: number;
  candidateFrames: FrameResult[];
  selectedAnswer: string;
  webGrounding: boolean;
  webSearchAvailable: boolean | null;
  visualVerificationAvailable: boolean | null;
  onAnalyze: () => void;
  onWebGroundingChange: (enabled: boolean) => void;
  onChooseAnswer: (answer: string, evidenceId?: string) => void;
  onOpenFrame: (submitKeyframeId: string) => void;
  /** Model behind the visual passes, from /api/health (before any analysis). */
  visionModel?: string | null;
  /** "deepseek" or "nvila": which backend to point at when it is offline. */
  visionBackend?: string | null;
}

export function QaAssistPanel(props: Props) {
  const vision = qaVisionLabel(props.analysis?.model ?? props.visionModel);
  return (
    <div className="panel qa-assist" data-testid="qa-assist-panel">
      <div className="qa-assist-head">
        <div>
          <h3>QA copilot · {vision === "DeepSeek" ? "DeepSeek V4.1 Flash" : vision}</h3>
          <div className="qa-method">hotspot prediction → answer options → human verification</div>
        </div>
        {props.analysis && <span className="badge image_pe">{props.analysis.mode}</span>}
      </div>

      {props.available === false && (
        <div className="qa-service-warn">
          {props.visionBackend === "nvila" ? (
            <>Worker offline. Start the NVILA Colab worker and configure <span className="mono">NVILA_BASE_URL</span>.</>
          ) : (
            <>Visual QA unavailable. Set <span className="mono">DEEPSEEK_API_KEY</span> in the backend configuration.</>
          )}
        </div>
      )}

      <label className="qa-grounding-toggle">
        <input
          type="checkbox"
          checked={props.webGrounding}
          onChange={(event) => props.onWebGroundingChange(event.target.checked)}
        />
        <span>DeepSeek web search · auto when knowledge is missing</span>
        {props.webSearchAvailable === false && <span className="qa-grounding-off">no API key</span>}
        {props.webGrounding && props.visualVerificationAvailable === false && (
          <span className="qa-grounding-off">pass 3 offline</span>
        )}
      </label>

      {props.candidateFrames.length > 0 && (
        <>
          <div className="qa-section-label">Input candidates · selected C01 → top-video priority</div>
          <div className="qa-input-candidates" data-testid="qa-input-candidates">
            {props.candidateFrames.map((frame, index) => (
              <button
                key={frame.submit_keyframe_id}
                className="qa-input-candidate"
                title={`${frame.submit_keyframe_id} · retrieval ${frame.score.toFixed(4)}`}
                onClick={() => props.onOpenFrame(frame.submit_keyframe_id)}
              >
                <img src={frame.keyframe_url} alt={frame.submit_keyframe_id} />
                <span>C{String(index + 1).padStart(2, "0")}</span>
                <span className="mono">{frame.video_id}</span>
              </button>
            ))}
          </div>
        </>
      )}

      <button
        className="btn primary qa-analyze-btn"
        data-testid="qa-analyze"
        disabled={props.loading || props.candidateCount === 0}
        onClick={props.onAnalyze}
      >
        {props.loading ? `${vision} is inspecting frames…` : `Analyze ${props.candidateCount} visual candidates`}
      </button>

      {props.error && <div className="dup-warn" data-testid="qa-analysis-error">{props.error}</div>}

      {props.analysis && (
        <div data-testid="qa-analysis">
          <div className="qa-run-meta mono">
            {props.analysis.model} · {((props.analysis.total_latency_ms ?? props.analysis.latency_ms) / 1000).toFixed(1)}s
            {props.analysis.cached ? " · cached" : ""}
          </div>

          {!props.analysis.answerable && (
            <div className="qa-service-warn">{vision} found insufficient evidence; broaden the query or candidate set.</div>
          )}

          <div className="qa-answer-list">
            {props.analysis.candidate_answers.map((candidate, index) => {
              const selected = candidate.answer === props.selectedAnswer;
              const evidenceId = candidate.supporting_frames[0]?.submit_keyframe_id;
              return (
                <button
                  key={`${candidate.answer}-${index}`}
                  className={`qa-answer-card ${selected ? "selected" : ""}`}
                  data-testid="qa-answer-option"
                  onClick={() => props.onChooseAnswer(candidate.answer, evidenceId)}
                >
                  <span className="qa-answer-rank">A{index + 1}</span>
                  <span className="qa-answer-copy">
                    <b>{candidate.answer}</b>
                    <span>{candidate.reason || "The model did not provide a verification reason."}</span>
                    <span className={`qa-answer-source ${candidate.source}`}>
                      {candidate.source === "web"
                        ? "Web grounded"
                        : candidate.source === "knowledge"
                          ? "Model knowledge (no web search)"
                          : candidate.source === "hybrid"
                            ? `${vision} + web`
                            : `${vision} visual`}
                    </span>
                    {candidate.visual_verification && (
                      <span className={`qa-verify-status ${candidate.visual_verification.status}`}>
                        {candidate.visual_verification.status === "supported"
                          ? `${vision} visually consistent · ${Math.round(candidate.visual_verification.visual_confidence * 100)}%`
                          : candidate.visual_verification.status === "contradicted"
                            ? `${vision} contradicted`
                            : candidate.visual_verification.status === "insufficient"
                              ? `${vision}: insufficient visual evidence`
                              : `${vision} verification unavailable`}
                      </span>
                    )}
                    <span className="qa-support-ids mono">
                      {candidate.supporting_frames.map((frame) => frame.submit_keyframe_id).join(" · ") || "no grounded frame"}
                    </span>
                  </span>
                  <span className="qa-confidence">{Math.round(candidate.confidence * 100)}%</span>
                </button>
              );
            })}
          </div>

          {props.analysis.candidate_answers.length === 0 && (
            <div className="qa-service-warn">No valid answer candidates; zero-confidence and placeholder outputs were removed.</div>
          )}

          {props.analysis.hotspots.length > 0 && (
            <>
              <div className="qa-section-label">Answer-bearing hotspots</div>
              <div className="qa-hotspots">
                {props.analysis.hotspots.map((hotspot, index) => (
                  <button
                    key={hotspot.candidate_id}
                    className="qa-hotspot"
                    data-testid="qa-hotspot"
                    title={hotspot.answer_support}
                    onClick={() => props.onOpenFrame(hotspot.submit_keyframe_id)}
                  >
                    <img src={hotspot.keyframe_url} alt={hotspot.submit_keyframe_id} />
                    <span>H{index + 1} · {Math.round(hotspot.relevance * 100)}%</span>
                    <span className="mono">{hotspot.video_id} · {formatTime(hotspot.pts_time)}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          {props.analysis.uncertainty && (
            <div className="qa-uncertainty">Verify: {props.analysis.uncertainty}</div>
          )}

          {props.analysis.web_grounding?.attempted && (
            <div className="qa-web-grounding" data-testid="qa-web-grounding">
              <div className="qa-section-label">
                Web search · {props.analysis.web_grounding.used ? "grounded" : "no usable answer"}
                {props.analysis.web_grounding.model ? ` · ${props.analysis.web_grounding.model}` : ""}
              </div>
              {props.analysis.web_grounding.queries.length > 0 && (
                <div className="qa-web-queries mono">
                  {props.analysis.web_grounding.queries.join(" · ")}
                </div>
              )}
              {props.analysis.web_grounding.visual_verification?.uncertainty && (
                <div className="qa-uncertainty">
                  Pass 3: {props.analysis.web_grounding.visual_verification.uncertainty}
                </div>
              )}
              {props.analysis.web_grounding.visual_verification?.attempted && (
                <div className="qa-pass3-summary" data-testid="qa-pass3-verification">
                  {vision} pass 3 · {props.analysis.web_grounding.visual_verification?.used ? "checked" : "no usable verdict"}
                  {props.analysis.web_grounding.visual_verification?.model
                    ? ` · ${props.analysis.web_grounding.visual_verification.model}` : ""}
                  {(props.analysis.web_grounding.visual_verification?.rejected_answers.length ?? 0) > 0
                    ? ` · rejected: ${props.analysis.web_grounding.visual_verification?.rejected_answers.join(", ")}` : ""}
                </div>
              )}
              <div className="qa-web-sources">
                {props.analysis.web_grounding.sources.map((source) => (
                  <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                    {source.title || new URL(source.url).hostname}
                  </a>
                ))}
              </div>
              {props.analysis.web_grounding.search_suggestions_html && (
                <iframe
                  className="qa-search-suggestions"
                  title="Web search suggestions"
                  sandbox="allow-popups allow-popups-to-escape-sandbox"
                  srcDoc={props.analysis.web_grounding.search_suggestions_html}
                />
              )}
            </div>
          )}

          {props.analysis.warnings.length > 0 && (
            <div className="qa-analysis-warnings">
              {props.analysis.warnings.map((warning) => <div key={warning}>{warning}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
