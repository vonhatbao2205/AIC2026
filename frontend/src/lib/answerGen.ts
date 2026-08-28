/** Bulk answer generation: fill the submission table from the answer generator.
 *
 * The Submission tab is where the 100-answer lists live, so this is where the
 * generator is driven from. One question at a time, sequentially: five operators
 * share one backend and one PE encoder, and firing 40 searches at once would
 * queue behind each other anyway while making the progress bar useless.
 */
import { api, ApiError } from "../api/client";
import type {
  AnswerGenParams,
  GeneratedAnswer,
  ImageEmbeddingModel,
  RetrievalDatabase,
  SearchScope,
} from "../api/types";
import { imageModelsForSearch } from "./imageModels";
import { KIND_QUERY_TYPE, type ImportedQuestion } from "./questions";
import { MAX_ROWS_PER_QUESTION, frameKey, type SubmissionRow } from "./submission";

/** Every index a profile can answer with; `imageModelsForSearch` trims it to
 *  what the active profile actually has. */
const ALL_IMAGE_MODELS: ImageEmbeddingModel[] = ["pe", "qwen3_vl"];

export interface AnswerGenRequest {
  questions: ImportedQuestion[];
  retrievalDatabase: RetrievalDatabase;
  /** Image indices the generator searches. Defaults to every index the profile
   *  has, so a generated list is built from the same evidence a two-model
   *  console search would surface, not from PE alone. */
  imageModels?: ImageEmbeddingModel[];
  scope?: SearchScope;
  /** Answers the question should end up with, counting the kept ones. */
  limit: number;
  params?: AnswerGenParams;
  /** Rows a person picked, which the generator must never replace.
   *
   *  They keep the head of the list — a frame somebody verified by eye belongs at
   *  rank 1, where being right is worth the whole query (`R@1` feeds all five
   *  cutoffs) rather than the 0.2 it earns at rank 100 — and they shrink the
   *  budget so the question still ends at 100 rows. */
  keptRows: (questionId: string) => SubmissionRow[];
  /** True when a question should be left alone entirely. */
  skip: (questionId: string) => boolean;
  /** Q&A text to write into that question's rows, when the operator has one. */
  answerText?: (questionId: string) => string;
}

export interface AnswerGenProgress {
  questionId: string;
  index: number;
  total: number;
  status: "running" | "done" | "failed" | "skipped";
  produced?: number;
  error?: string;
}

export interface AnswerGenOutcome {
  questionId: string;
  rows: Omit<SubmissionRow, "id">[];
  warnings: string[];
  error?: string;
  skipped?: boolean;
}

/** One question's generated answers, already shaped as submission rows. */
export async function generateForQuestion(
  question: ImportedQuestion,
  options: {
    retrievalDatabase: RetrievalDatabase;
    imageModels?: ImageEmbeddingModel[];
    scope?: SearchScope;
    limit: number;
    params?: AnswerGenParams;
    answerText?: string;
    /** Answers already at the head of the list, so the generator neither repeats
     *  them nor re-covers the instants they already cover. */
    taken?: SubmissionRow[];
    signal?: AbortSignal;
  },
): Promise<AnswerGenOutcome> {
  const limit = Math.max(1, Math.min(options.limit, MAX_ROWS_PER_QUESTION));
  const response = await api.generateAnswers(
    {
      retrieval_database: options.retrievalDatabase,
      // Normalised on the way out, so no caller can produce an invalid request:
      // a BTC one must never name qwen3_vl (those keyframes have no Qwen
      // vectors and the API rejects it), and an empty list must never be sent.
      image_models: imageModelsForSearch(
        options.retrievalDatabase,
        options.imageModels ?? ALL_IMAGE_MODELS,
      ),
      query: question.text,
      query_type_hint: KIND_QUERY_TYPE[question.kind],
      ...(options.scope ? { scope: options.scope } : {}),
      limit,
      // The statement is the authority on how many frames a TRAKE row carries;
      // the backend uses it to refuse chains that located fewer moments, which
      // would otherwise be written as rows the export rejects.
      ...(question.eventCount ? { event_count: question.eventCount } : {}),
      ...(options.taken?.length
        ? {
            taken: options.taken
              .filter((row) => row.videoId.trim() && row.frames.length)
              .map((row) => ({ video_id: row.videoId.trim(), frames: row.frames })),
          }
        : {}),
      ...(options.params ? { params: options.params } : {}),
      ...(options.answerText ? { answer_text: options.answerText } : {}),
    },
    options.signal,
  );
  return {
    questionId: question.id,
    warnings: response.warnings ?? [],
    rows: response.answers.map((answer) => toRow(question, answer, options.retrievalDatabase)),
  };
}

function toRow(
  question: ImportedQuestion,
  answer: GeneratedAnswer,
  retrievalDatabase: RetrievalDatabase,
): Omit<SubmissionRow, "id"> {
  return {
    questionId: question.id,
    videoId: answer.video_id,
    frames: answer.frames,
    answer: answer.answer ?? "",
    // The generator owns these: regenerating replaces exactly this set and
    // leaves every row a person picked untouched.
    source: "generated",
    keyframeIds: answer.keyframe_ids,
    ptsTimes: answer.pts_times,
    retrievalDatabase,
  };
}

/** Run every question in order, reporting progress as it goes.
 *
 *  Never throws for a single failed question: a dead channel on question 7 must
 *  not cost the other 39 their answers. Failures come back in the outcome list. */
export async function generateAll(
  request: AnswerGenRequest,
  onProgress: (progress: AnswerGenProgress) => void,
  signal?: AbortSignal,
): Promise<AnswerGenOutcome[]> {
  const outcomes: AnswerGenOutcome[] = [];
  const total = request.questions.length;
  for (let index = 0; index < total; index += 1) {
    if (signal?.aborted) break;
    const question = request.questions[index];
    const kept = request.keptRows(question.id);
    const budget = Math.min(request.limit, MAX_ROWS_PER_QUESTION) - kept.length;
    if (request.skip(question.id) || budget <= 0) {
      onProgress({ questionId: question.id, index, total, status: "skipped" });
      outcomes.push({ questionId: question.id, rows: [], warnings: [], skipped: true });
      continue;
    }
    onProgress({ questionId: question.id, index, total, status: "running" });
    try {
      const outcome = await generateForQuestion(question, {
        retrievalDatabase: request.retrievalDatabase,
        imageModels: request.imageModels,
        scope: request.scope,
        limit: budget,
        params: request.params,
        answerText: request.answerText?.(question.id),
        taken: kept,
        signal,
      });
      // The backend already drops anything the kept rows cover, but the CSV is
      // what gets graded: a duplicate row here is a position spent twice.
      const seen = new Set(kept.map(frameKey));
      outcome.rows = outcome.rows.filter((row) => !seen.has(frameKey(row))).slice(0, budget);
      outcomes.push(outcome);
      onProgress({
        questionId: question.id,
        index,
        total,
        status: "done",
        produced: outcome.rows.length,
      });
    } catch (error) {
      if (signal?.aborted) break;
      const message =
        error instanceof ApiError
          ? `${error.status}: ${error.message}`
          : error instanceof Error
            ? error.message
            : "lỗi không rõ";
      outcomes.push({ questionId: question.id, rows: [], warnings: [], error: message });
      onProgress({ questionId: question.id, index, total, status: "failed", error: message });
    }
  }
  return outcomes;
}
