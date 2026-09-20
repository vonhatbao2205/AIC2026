import { describe, expect, it } from "vitest";
import type { QueryType } from "../api/types";
import type { ImportedQuestion } from "../lib/questions";
import {
  parseQuestionEntry,
  parseQuestionPack,
  trakeEventCount,
  kindForQueryType,
} from "../lib/questions";
import {
  SubmissionFormatError,
  auditSubmission,
  buildSubmissionFiles,
  buildSubmissionZip,
  buildSubmissionZipBytes,
  csvField,
  findDuplicate,
  MAX_ANSWER_LENGTH,
  MAX_ROWS_PER_QUESTION,
  qaAnswerMissingEverywhere,
  questionCsv,
  rowKey,
  rowToCsvLine,
  validateRows,
  type SubmissionRow,
} from "../lib/submission";
import { buildZipBytes, crc32, ZipError } from "../lib/zip";
import {
  parseAnswerCsv,
  parseCsvLine,
  parseSubmissionPack,
  planImport,
} from "../lib/submissionImport";
import {
  canonicalQuestions,
  normalizeQuestions,
  packHash,
  packSummary,
  recordsToQuestions,
} from "../lib/questionPack";
import {
  MISSING_TABLE_HINT,
  describeSupabaseError,
  isMissingTableError,
  mergeRow,
  newRowId,
  nextCreatedAt,
  recordToRow,
  rowToRecord,
  sortRows,
} from "../lib/sharedSubmission";

const entry = (name: string, text = "nội dung") => ({ name, text });

function row(over: Partial<SubmissionRow> = {}): SubmissionRow {
  return { id: "r1", questionId: "q", videoId: "L01_V028", frames: [3450], answer: "", ...over };
}

describe("query pack parsing", () => {
  it("reads the type and order out of the file name", () => {
    expect(parseQuestionEntry(entry("query-p1-15-qa.txt"))).toMatchObject({
      id: "query-p1-15-qa",
      order: 15,
      kind: "qa",
      queryType: "QA",
    });
    expect(parseQuestionEntry(entry("query-p1-1-kis.txt"))).toMatchObject({ kind: "kis", queryType: "T-KIS" });
    expect(parseQuestionEntry(entry("query-p1-4-trake.txt"))).toMatchObject({ kind: "trake", queryType: "TRAKE" });
  });

  it("opens a -vkis question as V-KIS but submits it in the KIS shape", () => {
    const question = parseQuestionEntry(entry("query-p1-9-vkis.txt"));
    expect(question).toMatchObject({ queryType: "V-KIS", kind: "kis" });
    expect(kindForQueryType("V-KIS")).toBe("kis");
  });

  it("ignores macOS resource forks and unrelated files", () => {
    expect(parseQuestionEntry(entry("__MACOSX/query-p1-1-kis.txt"))).toBeNull();
    expect(parseQuestionEntry(entry("._query-p1-1-kis.txt"))).toBeNull();
    expect(parseQuestionEntry(entry("readme.txt"))).toBeNull();
  });

  it("counts TRAKE events from the E1..EN markers", () => {
    expect(
      trakeEventCount("E1: bột vào tô.\nE2: chạm dầu.\nE3: rời chảo.\nE4: nằm trên dĩa."),
    ).toBe(4);
    expect(trakeEventCount("không có sự kiện nào")).toBeNull();
  });

  it("counts the descriptions, not the numbers written on them", () => {
    // Regression, from a real pack: the labels run E1, E2, E2, E4 but there are
    // four moments to find. Counting distinct indices gave 3, and every correct
    // four-frame row was then rejected at export for being the wrong width.
    const misnumbered = [
      "Trong đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế:",
      "E1: Khoảnh khắc đầu tiên thấy cắt nấm.",
      "E2: Khoảnh khắc đầu tiên cắt củ năng.",
      "E2: Khoảnh khắc đầu tiên cắt đậu hủ.",
      "E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa và thấy lửa bắt đầu xuất hiện",
    ].join("\n");
    expect(trakeEventCount(misnumbered)).toBe(4);

    // A skipped number is the mirror image: three descriptions, three frames —
    // so the highest index would have been just as wrong as the distinct count.
    expect(trakeEventCount("E1: a.\nE2: b.\nE4: c.")).toBe(3);
  });

  it("accepts the marker spellings the backend event splitter accepts", () => {
    expect(trakeEventCount("Sự kiện 1: chạy đà.\nSự kiện 2: giậm nhảy.")).toBe(2);
    expect(trakeEventCount("Event 1) take-off.\nEvent 2) landing.")).toBe(2);
    expect(trakeEventCount("E1 - vào bếp.\nE2 - ra món.")).toBe(2);
    // Two events written on one line still count as two.
    expect(trakeEventCount("E1: bột vào tô. E2: chạm dầu.")).toBe(2);
  });

  it("reads markers that carry no punctuation after the number", () => {
    // Regression, from a real organiser pack: "E1 Khoảnh khắc…" with a space and
    // no colon. Requiring punctuation matched nothing, the count came back null,
    // and the width check below quietly switched itself off on the one question
    // whose rows are the easiest to get wrong.
    const statement = [
      "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ.",
      "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.",
      "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ (thời điểm đâu tiên các chân của lân đặt trên trụ sau khi xoay).",
      "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.",
    ].join("\n");
    expect(trakeEventCount(statement)).toBe(3);
    expect(trakeEventCount("Sự kiện 1 chạy đà.\nSự kiện 2 giậm nhảy.")).toBe(2);
  });

  it("does not mistake prose for event markers", () => {
    // Nothing here is a marker: no "E<n>" follows a word boundary, and a bare
    // numbered list is deliberately not counted (null skips the width check
    // rather than blocking the export on a guess).
    expect(trakeEventCount("Cảnh quay phase 2: người đàn ông bước vào lúc 18:30.")).toBeNull();
    expect(trakeEventCount("1) chạy đà.\n2) giậm nhảy.\n3) tiếp đất.")).toBeNull();
    // One marker is not a sequence.
    expect(trakeEventCount("E1: chỉ có một khoảnh khắc.")).toBeNull();
  });

  it("keeps the organiser's numeric order, not lexicographic order", () => {
    const pack = parseQuestionPack([
      entry("query-p1-25-kis.txt"),
      entry("query-p1-2-kis.txt"),
      entry("query-p1-10-kis.txt"),
    ]);
    expect(pack.map((q) => q.order)).toEqual([2, 10, 25]);
  });
});

describe("csv shaping", () => {
  it("quotes only when the spec requires it", () => {
    expect(csvField("Năm người")).toBe("Năm người");
    expect(csvField("5")).toBe("5");
    expect(csvField("Màu đỏ, rất đẹp")).toBe('"Màu đỏ, rất đẹp"');
    expect(csvField('Anh ấy nói "Tuyệt vời"')).toBe('"Anh ấy nói ""Tuyệt vời"""');
    expect(csvField("Dòng 1\nDòng 2")).toBe('"Dòng 1\nDòng 2"');
    expect(csvField(" giữ khoảng trắng ")).toBe('" giữ khoảng trắng "');
  });

  it("writes one line per query shape", () => {
    expect(rowToCsvLine(row(), "kis")).toBe("L01_V028,3450");
    expect(rowToCsvLine(row({ answer: "Năm người" }), "qa")).toBe("L01_V028,3450,Năm người");
    expect(rowToCsvLine(row({ videoId: "L10_V001", frames: [1200, 1850, 2100, 2450] }), "trake")).toBe(
      "L10_V001,1200,1850,2100,2450",
    );
  });

  it("joins rows with CRLF and no header", () => {
    const csv = questionCsv([row({ id: "a" }), row({ id: "b", frames: [5555] })], "kis");
    expect(csv).toBe("L01_V028,3450\r\nL01_V028,5555");
    expect(csv.startsWith("video")).toBe(false);
  });
});

describe("duplicate detection", () => {
  it("treats the same frame as a repeat for KIS", () => {
    const rows = [row({ id: "a" })];
    expect(findDuplicate(rows, { questionId: "q", videoId: "L01_V028", frames: [3450], answer: "" }, "kis")).toBeTruthy();
    expect(findDuplicate(rows, { questionId: "q", videoId: "L01_V028", frames: [3451], answer: "" }, "kis")).toBeNull();
  });

  it("lets the same QA frame carry a different answer", () => {
    const rows = [row({ id: "a", answer: "Năm người" })];
    const same = { questionId: "q", videoId: "L01_V028", frames: [3450], answer: "năm  NGƯỜI" };
    const other = { questionId: "q", videoId: "L01_V028", frames: [3450], answer: "Sáu người" };
    expect(findDuplicate(rows, same, "qa")).toBeTruthy(); // only whitespace/case differ
    expect(findDuplicate(rows, other, "qa")).toBeNull();
  });

  it("scopes duplicates to one question", () => {
    const rows = [row({ id: "a", questionId: "q1" })];
    expect(findDuplicate(rows, { questionId: "q2", videoId: "L01_V028", frames: [3450], answer: "" }, "kis")).toBeNull();
  });

  it("keys a TRAKE row on the whole ordered sequence", () => {
    expect(rowKey({ videoId: "V", frames: [1, 2, 3], answer: "" }, "trake")).not.toBe(
      rowKey({ videoId: "V", frames: [1, 3, 2], answer: "" }, "trake"),
    );
  });
});

describe("row validation", () => {
  const trake = { id: "q", order: 4, kind: "trake" as const, queryType: "TRAKE" as const, text: "", eventCount: 4 };
  const qa = { id: "q", order: 1, kind: "qa" as const, queryType: "QA" as const, text: "", eventCount: null };

  it("flags a TRAKE row with the wrong number of events", () => {
    const problems = validateRows([row({ frames: [1, 2, 3] })], trake);
    expect(problems.some((p) => p.message.includes("requires exactly 4 frames"))).toBe(true);
  });

  it("flags TRAKE frames that do not increase", () => {
    const problems = validateRows([row({ frames: [10, 9, 30, 40] })], trake);
    expect(problems.some((p) => p.message.includes("increasing chronological order"))).toBe(true);
  });

  it("flags an over-long or empty QA answer", () => {
    expect(validateRows([row({ answer: "" })], qa).some((p) => p.message.includes("missing answer"))).toBe(true);
    expect(
      validateRows([row({ answer: "x".repeat(101) })], qa).some((p) => p.message.includes("exceeds 100")),
    ).toBe(true);
  });

  it("says so when the event count could not be read, instead of going quiet", () => {
    const unknown = { ...trake, eventCount: null };
    const problems = validateRows([row({ frames: [1, 2, 3] })], unknown);
    const notice = problems.find((p) => p.message.includes("could not determine the event count"));
    expect(notice?.severity).toBe("warning");
  });

  it("blocks TRAKE rows of differing width when the event count is unknown", () => {
    // One frame per event means every row of a question carries the same count.
    // Which of them is wrong cannot be known, but that they disagree can.
    const unknown = { ...trake, eventCount: null };
    const problems = validateRows(
      [row({ id: "a", frames: [1, 2, 3] }), row({ id: "b", frames: [4, 5] })],
      unknown,
    );
    const blocker = problems.find((p) => p.message.includes("inconsistent"));
    expect(blocker?.severity).toBe("error");
    expect(blocker?.message).toContain("2 / 3");
    // Rows that agree are left alone — the count is unverifiable, not wrong.
    expect(
      validateRows([row({ id: "a", frames: [1, 2] }), row({ id: "b", frames: [4, 5] })], unknown)
        .some((p) => p.severity === "error"),
    ).toBe(false);
  });

  it("flags a video name that still has its .mp4 extension", () => {
    const problems = validateRows([row({ videoId: "L01_V028.mp4" })], qa);
    expect(problems.some((p) => p.message.includes(".mp4"))).toBe(true);
  });
});

describe("zip writing", () => {
  it("computes the standard CRC-32", () => {
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
  });

  it("emits a readable archive with the required submission/ folder", () => {
    const bytes = buildZipBytes([{ name: "submission/query-p1-1-kis.csv", text: "L01_V028,25300" }]);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(0, true)).toBe(0x04034b50); // local file header
    expect(view.getUint32(bytes.length - 22, true)).toBe(0x06054b50); // end of central directory
    const text = new TextDecoder().decode(bytes);
    expect(text).toContain("submission/query-p1-1-kis.csv");
    expect(text).toContain("L01_V028,25300");
  });
});

describe("shared submission records", () => {
  const record = {
    id: "11111111-1111-4111-8111-111111111111",
    room: "aic26",
    session_id: "22222222-2222-4222-8222-222222222222",
    question_id: "query-p1-4-trake",
    retrieval_database: "infoshotpp",
    video_id: "L27_V013",
    frames: [1200, 1850],
    keyframe_ids: ["L27/L27_V013/042", null],
    pts_times: [48.0, null],
    answer: "",
    submitted_by: "Bao",
    source: "submit",
    revision: 3,
    created_at: "2026-08-18T10:00:00Z",
    updated_at: "2026-08-18T10:05:00Z",
  };

  it("round-trips a row through the wire shape", () => {
    const row = recordToRow(record);
    expect(row).toMatchObject({
      id: record.id,
      questionId: "query-p1-4-trake",
      videoId: "L27_V013",
      frames: [1200, 1850],
      keyframeIds: ["L27/L27_V013/042", null],
      ptsTimes: [48, null],
      retrievalDatabase: "infoshotpp",
      submittedBy: "Bao",
      revision: 3,
      syncState: "synced",
    });
    const back = rowToRecord(row, "aic26", "Bao");
    expect(back.frames).toEqual(record.frames);
    expect(back.keyframe_ids).toEqual(record.keyframe_ids);
    expect(back.pts_times).toEqual(record.pts_times);
  });

  it("keeps the preview arrays aligned with the frames they describe", () => {
    // A TRAKE row edited frame-by-frame can reach the wire with short arrays;
    // index i must keep meaning frame i or a preview would show another event.
    const row: SubmissionRow = {
      id: "r", questionId: "q", videoId: "V", frames: [10, 20, 30], answer: "",
      keyframeIds: ["a"], ptsTimes: [],
    };
    const sent = rowToRecord(row, "aic26", "me");
    expect(sent.keyframe_ids).toEqual(["a", null, null]);
    expect(sent.pts_times).toEqual([null, null, null]);
  });

  it("never lets a stale reply roll a row back", () => {
    const older: SubmissionRow = { id: "r", questionId: "q", videoId: "V", frames: [1], answer: "", revision: 2 };
    const newer: SubmissionRow = { ...older, frames: [9], revision: 5 };
    expect(mergeRow([newer], older)[0].frames).toEqual([9]);
    expect(mergeRow([older], newer)[0].frames).toEqual([9]);
    expect(mergeRow([], newer)).toHaveLength(1);
  });

  it("orders rows by creation, so editing one does not re-rank the CSV", () => {
    const rows: SubmissionRow[] = [
      { id: "b", questionId: "q", videoId: "V", frames: [2], answer: "", createdAt: "2026-01-02", updatedAt: "2026-01-02" },
      { id: "a", questionId: "q", videoId: "V", frames: [1], answer: "", createdAt: "2026-01-01", updatedAt: "2026-01-09" },
    ];
    expect(sortRows(rows).map((row) => row.id)).toEqual(["a", "b"]);
  });

  it("generates ids the uuid column accepts", () => {
    expect(newRowId()).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  });
});

describe("answer order survives the store", () => {
  const at = (createdAt: string, id: string): SubmissionRow => ({
    id, questionId: "q", videoId: "L01_V028", frames: [1], answer: "", createdAt,
  });

  it("hands out a strictly later stamp every time", () => {
    // A hundred rows inserted together used to share one stamp, which dropped
    // the whole block onto the random-uuid tiebreak.
    const stamps = Array.from({ length: 200 }, () => nextCreatedAt());
    expect(new Set(stamps).size).toBe(stamps.length);
    expect([...stamps].sort()).toEqual(stamps);
  });

  it("keeps a generated batch in the order it was generated", () => {
    const batch = Array.from({ length: 100 }, (_, rank) => at(nextCreatedAt(), `row-${rank}`));
    // Shuffled by id, exactly the way the uuid tiebreak used to shuffle them.
    const scrambled = [...batch].sort((a, b) => a.id.localeCompare(b.id));
    expect(sortRows(scrambled).map((row) => row.id)).toEqual(batch.map((row) => row.id));
  });

  it("orders by instant, not by how the timestamp happens to be spelled", () => {
    // PostgREST drops the fraction when the microseconds are zero, so the same
    // moment can come back as `…T19:04:00+00:00`. Compared as text that sorts
    // AFTER `…T19:04:00.500Z` — half a second later than it actually is.
    const early = "2026-08-20T19:04:00+00:00";
    const late = "2026-08-20T19:04:00.500Z";
    expect(early.localeCompare(late)).toBeGreaterThan(0); // the trap
    expect(Date.parse(early)).toBeLessThan(Date.parse(late)); // the truth
    expect(sortRows([at(late, "b"), at(early, "a")]).map((row) => row.id)).toEqual(["a", "b"]);
  });

  it("normalises a server timestamp to one canonical spelling", () => {
    const row = recordToRow({
      id: "a", room: "aic26", session_id: null, question_id: "q",
      retrieval_database: "btc", video_id: "L01_V028", frames: [1],
      keyframe_ids: [null], pts_times: [null], answer: "", submitted_by: "bao",
      source: "generated", revision: 1, created_at: "2026-08-20T19:04:00.123456+00:00",
    });
    expect(row.createdAt).toBe("2026-08-20T19:04:00.123Z");
  });

  it("sends created_at so a bulk insert is not stamped all at once", () => {
    // `created_at` defaults to now(), which is the TRANSACTION time and so is
    // identical for every row of one multi-row INSERT.
    const row = at("2026-08-20T19:04:00.123Z", "a");
    expect(rowToRecord(row, "aic26", "bao", null).created_at).toBe("2026-08-20T19:04:00.123Z");
  });

  it("round-trips a batch through the record shape without reordering", () => {
    const batch = Array.from({ length: 20 }, (_, rank) => at(nextCreatedAt(), `row-${rank}`));
    const back = batch.map((row) =>
      recordToRow({
        ...rowToRecord(row, "aic26", "bao", null),
        id: row.id,
        room: "aic26",
        // PostgREST's rendering of the very same instant.
        created_at: new Date(row.createdAt!).toISOString().replace("Z", "+00:00"),
      }),
    );
    const scrambled = [...back].sort((a, b) => a.id.localeCompare(b.id));
    expect(sortRows(scrambled).map((row) => row.id)).toEqual(batch.map((row) => row.id));
  });
});

describe("supabase error reporting", () => {
  it("reads a PostgREST error object instead of stringifying it", () => {
    // Regression: PostgREST rejects with a plain object, so the usual
    // `instanceof Error` check fell through to String() and showed
    // "[object Object]" — an error message that tells an operator nothing.
    const postgrest = {
      message: "Could not find the table 'public.submissions' in the schema cache",
      details: null,
      hint: null,
      code: "PGRST205",
    };
    expect(describeSupabaseError(postgrest)).toContain("Could not find the table");
    expect(describeSupabaseError(postgrest)).toContain("PGRST205");
    expect(describeSupabaseError(postgrest)).not.toContain("[object Object]");
  });

  it("still handles Errors, strings and shapeless values", () => {
    expect(describeSupabaseError(new Error("boom"))).toBe("boom");
    expect(describeSupabaseError("plain")).toBe("plain");
    expect(describeSupabaseError({ weird: 1 })).toBe('{"weird":1}');
    expect(describeSupabaseError(null)).toBe("Unknown error");
  });

  it("recognises the un-run migration and says what to do about it", () => {
    expect(isMissingTableError({ code: "PGRST205", message: "..." })).toBe(true);
    expect(isMissingTableError({ code: "42P01", message: "relation does not exist" })).toBe(true);
    expect(
      isMissingTableError({ message: "Could not find the table 'public.submissions' in the schema cache" }),
    ).toBe(true);
    expect(isMissingTableError({ code: "23505", message: "duplicate key" })).toBe(false);
    expect(MISSING_TABLE_HINT).toContain("001_shared_submission.sql");
  });
});

describe("question pack as shared state", () => {
  const pack = (over: Partial<{ order: number; text: string }> = {}) => [
    { id: "query-p1-1-kis", order: 1, kind: "kis" as const, queryType: "T-KIS" as const, text: "a", eventCount: null },
    (() => {
      const text = over.text ?? "E1: x\nE2: y";
      return {
        id: "query-p1-4-trake", order: over.order ?? 4, kind: "trake" as const,
        // Derived, like a real import — a helper that hardcoded the count would
        // hide exactly the bug the round-trip is supposed to catch.
        queryType: "TRAKE" as const, text, eventCount: trakeEventCount(text),
      };
    })(),
  ];

  it("fingerprints the same pack identically whatever order it arrives in", () => {
    // Two machines must agree, or each would think the other published an
    // update and they would re-download forever.
    const forward = pack();
    const reversed = [...pack()].reverse();
    expect(packHash(forward)).toBe(packHash(reversed));
  });

  it("changes the fingerprint when any question changes", () => {
    expect(packHash(pack())).not.toBe(packHash(pack({ text: "E1: x\nE2: z" })));
    expect(packHash(pack())).not.toBe(packHash(pack({ order: 5 })));
    expect(packHash(pack())).not.toBe(packHash(pack().slice(0, 1)));
  });

  it("round-trips through the wire shape the RPC receives", () => {
    const canonical = canonicalQuestions(pack());
    expect(canonical.map((q) => q.question_id)).toEqual(["query-p1-1-kis", "query-p1-4-trake"]);
    expect(canonical[1]).toMatchObject({ ordinal: 4, kind: "trake", query_type: "TRAKE", event_count: 2 });
    expect(recordsToQuestions(canonical)).toEqual(pack());
  });

  it("re-derives the TRAKE event count instead of trusting the stored column", () => {
    // A pack published before the counting fix carries the stale number. The
    // count is a pure function of the statement, so reading it back recomputes:
    // otherwise the whole team would validate rows against 3 for the rest of the
    // round, with no way to correct it short of re-publishing the pack.
    const stale = canonicalQuestions(pack({ text: "E1: a.\nE2: b.\nE2: c.\nE4: d." }));
    expect(stale[1].event_count).toBe(4);
    stale[1].event_count = 3;
    expect(recordsToQuestions(stale)[1].eventCount).toBe(4);
  });

  it("re-derives a stale event count held only in the local cache", () => {
    // The path that kept the wrong number on screen after the parser was fixed.
    // `refresh()` skips re-downloading a pack whose fingerprint it already holds,
    // so the cached copy is the only thing that ever renders — nothing on that
    // machine would recompute the count, for the rest of the round.
    const cached = [
      {
        id: "query-p1-18-trake",
        order: 18,
        kind: "trake" as const,
        queryType: "TRAKE" as const,
        text: "E1: cắt nấm.\nE2: cắt củ năng.\nE2: cắt đậu hủ.\nE4: chảo lên bếp.",
        eventCount: 3,
      },
    ];
    expect(normalizeQuestions(cached)[0].eventCount).toBe(4);
    // Non-TRAKE questions are untouched, and an already-correct pack is returned
    // as-is so a reload does not churn every object identity.
    const fresh = pack();
    expect(normalizeQuestions(fresh)).toEqual(fresh);
  });

  it("summarises a pack for the publish confirmation", () => {
    expect(packSummary(pack())).toEqual({ total: 2, byKind: { kis: 1, qa: 0, trake: 1 } });
  });
});

describe("Q&A answer text", () => {
  const question = {
    id: "query-p1-2-qa", order: 2, kind: "qa" as const,
    queryType: "QA" as const, text: "xã này tên là gì?", eventCount: null,
  };
  const row = (id: string, answer: string): SubmissionRow => ({
    id, questionId: question.id, videoId: "L30_V072", frames: [1776], answer,
  });

  it("states a missing answer once for the question, not once per row", () => {
    // The generator fills a hundred frames and the text is one decision, so the
    // usual state is a hundred rows sharing one blank. Reported per row that is
    // a hundred identical errors burying everything else in the panel.
    const rows = Array.from({ length: 100 }, (_, i) => row(`r${i}`, ""));
    const problems = validateRows(rows, question);
    const answerProblems = problems.filter((p) => p.message.includes("answer"));
    expect(answerProblems).toHaveLength(1);
    expect(answerProblems[0].message).toContain("all 100 rows will be blocked");
    // Still blocks the export — the organisers require an answer.
    expect(answerProblems[0].severity).toBe("error");
    expect(auditSubmission([question], rows).errors.length).toBe(1);
  });

  it("itemises per row once some rows do have an answer", () => {
    // Now WHICH rows are blank is real information rather than one repeated
    // sentence, so the collapse stops.
    const rows = [row("a", "5"), row("b", ""), row("c", "")];
    const messages = validateRows(rows, question)
      .filter((p) => p.message.includes("answer"))
      .map((p) => p.message);
    expect(messages).toEqual(["row 2: missing answer", "row 3: missing answer"]);
  });

  it("knows when a question is missing its answer everywhere", () => {
    expect(qaAnswerMissingEverywhere([row("a", ""), row("b", "")], question)).toBe(true);
    expect(qaAnswerMissingEverywhere([row("a", ""), row("b", "5")], question)).toBe(false);
    // A single blank row is already legible as one error; nothing to collapse.
    expect(qaAnswerMissingEverywhere([row("a", "")], question)).toBe(false);
    // Never applies outside Q&A: no other row shape carries an answer at all.
    const trake = { ...question, kind: "trake" as const, queryType: "TRAKE" as const };
    expect(qaAnswerMissingEverywhere([row("a", ""), row("b", "")], trake)).toBe(false);
  });

  it("still rejects an over-long answer per row", () => {
    const rows = [row("a", "x".repeat(MAX_ANSWER_LENGTH + 1))];
    expect(validateRows(rows, question).map((p) => p.message)).toContain(
      `row 1: answer length ${MAX_ANSWER_LENGTH + 1} exceeds ${MAX_ANSWER_LENGTH} characters`,
    );
  });
});

describe("importing an exported submission.zip", () => {
  const questions: ImportedQuestion[] = [
    { id: "query-p1-1-kis", order: 1, kind: "kis", queryType: "T-KIS", text: "a", eventCount: null },
    { id: "query-p1-2-qa", order: 2, kind: "qa", queryType: "QA", text: "b", eventCount: null },
    { id: "query-p1-3-trake", order: 3, kind: "trake", queryType: "TRAKE", text: "c", eventCount: 2 },
  ];
  const row = (over: Partial<SubmissionRow>): SubmissionRow => ({
    id: over.id ?? "r", questionId: "query-p1-1-kis", videoId: "L01_V028",
    frames: [1200], answer: "", ...over,
  });

  it("reads back exactly what the exporter wrote", () => {
    // The round trip is the specification: anything the writer can emit, the
    // reader has to take — including an answer holding the delimiter.
    const rows = [
      row({ id: "a", questionId: "query-p1-1-kis", frames: [25300] }),
      row({ id: "b", questionId: "query-p1-2-qa", frames: [3450], answer: "5, sáu" }),
      row({ id: "c", questionId: "query-p1-2-qa", frames: [3460], answer: 'nói "vâng"' }),
      row({ id: "d", questionId: "query-p1-3-trake", frames: [1200, 1850] }),
    ];
    const files = buildSubmissionFiles(questions, rows);
    const parsed = parseSubmissionPack(
      files.map((file) => ({ name: file.name, text: file.text })),
      questions,
    );
    expect(parsed.problems).toEqual([]);
    expect(parsed.totalRows).toBe(4);
    const flat = parsed.files.flatMap((file) => file.rows);
    expect(flat.map((r) => [r.questionId, r.videoId, r.frames, r.answer])).toEqual([
      ["query-p1-1-kis", "L01_V028", [25300], ""],
      ["query-p1-2-qa", "L01_V028", [3450], "5, sáu"],
      ["query-p1-2-qa", "L01_V028", [3460], 'nói "vâng"'],
      ["query-p1-3-trake", "L01_V028", [1200, 1850], ""],
    ]);
    // Somebody's earlier work: the generator must keep it, not replace it.
    expect(flat.every((r) => r.source === "manual")).toBe(true);
  });

  it("splits a quoted answer containing the delimiter", () => {
    expect(parseCsvLine('L01_V028,3450,"5, sáu"')).toEqual(["L01_V028", "3450", "5, sáu"]);
    expect(parseCsvLine('a,"say ""hi""",b')).toEqual(["a", 'say "hi"', "b"]);
    expect(parseCsvLine("a,,b")).toEqual(["a", "", "b"]);
  });

  it("needs the kind to tell a TRAKE row from an answered Q&A row", () => {
    // `L01_V028,1200,5` is both shapes at once; only the question says which.
    const line = "L01_V028,1200,5";
    expect(parseAnswerCsv(line, "q", "trake").rows[0].frames).toEqual([1200, 5]);
    const qa = parseAnswerCsv(line, "q", "qa").rows[0];
    expect(qa.frames).toEqual([1200]);
    expect(qa.answer).toBe("5");
  });

  it("falls back to the file-name suffix for a question not in the pack", () => {
    const parsed = parseSubmissionPack(
      [{ name: "submission/query-p9-9-trake.csv", text: "L01_V028,1200,1850" }],
      questions,
    );
    expect(parsed.files[0].kind).toBe("trake");
    expect(parsed.files[0].inPack).toBe(false);
    expect(parsed.orphanIds).toEqual(["query-p9-9-trake"]);
    expect(parsed.files[0].rows[0].frames).toEqual([1200, 1850]);
  });

  it("reports a malformed line instead of importing a broken row", () => {
    const parsed = parseSubmissionPack(
      [{ name: "submission/query-p1-1-kis.csv", text: "L01_V028,1200\n,1300\nL01_V028,abc\nL01_V028" }],
      questions,
    );
    const file = parsed.files[0];
    expect(file.rows).toHaveLength(1);
    expect(file.problems).toEqual([
      "query-p1-1-kis row 2: missing video name",
      'query-p1-1-kis row 3: "abc" is not a valid frame',
      "query-p1-1-kis row 4: no frames",
    ]);
  });

  it("ignores the empty CSV an unanswered question exports as", () => {
    const parsed = parseSubmissionPack(
      [{ name: "submission/query-p1-1-kis.csv", text: "" }],
      questions,
    );
    expect(parsed.files[0].rows).toEqual([]);
    expect(parsed.problems).toEqual([]);
  });

  it("says so when the zip is a question pack rather than answers", () => {
    // The two archives sit next to each other on disk and look identical.
    expect(() =>
      parseSubmissionPack([{ name: "query-p1-1-kis.txt", text: "bản tin" }], questions),
    ).toThrow(ZipError);
    expect(() =>
      parseSubmissionPack([{ name: "query-p1-1-kis.txt", text: "x" }], questions),
    ).toThrow(/QUESTION pack/);
  });

  it("caps a question at a hundred rows and says it did", () => {
    const text = Array.from({ length: 105 }, (_, i) => `L01_V028,${1000 + i}`).join("\r\n");
    const parsed = parseSubmissionPack(
      [{ name: "submission/query-p1-1-kis.csv", text }],
      questions,
    );
    expect(parsed.files[0].rows).toHaveLength(MAX_ROWS_PER_QUESTION);
    expect(parsed.files[0].problems[0]).toContain("105 rows");
  });

  it("plans the write against what is already on screen", () => {
    const parsed = parseSubmissionPack(
      [
        { name: "submission/query-p1-1-kis.csv", text: "L01_V028,25300" },
        { name: "submission/query-p1-2-qa.csv", text: "L01_V028,3450,5" },
      ],
      questions,
    );
    const held = [row({ id: "x", questionId: "query-p1-1-kis" })];

    // `fill` never destroys: a question already holding answers is left alone.
    const fill = planImport(parsed, held, "fill");
    expect(fill.write.map((f) => f.questionId)).toEqual(["query-p1-2-qa"]);
    expect(fill.replacedRows).toBe(0);

    // `replace` states the cost up front instead of discovering it afterwards.
    const replace = planImport(parsed, held, "replace");
    expect(replace.write.map((f) => f.questionId)).toEqual([
      "query-p1-1-kis",
      "query-p1-2-qa",
    ]);
    expect(replace.replacedRows).toBe(1);
  });
});

describe("export completeness and refusal", () => {
  const q = (id: string, kind: "kis" | "qa" | "trake", eventCount: number | null = null) => ({
    id,
    order: Number(id.split("-")[1]),
    kind,
    queryType: (kind === "qa" ? "QA" : kind === "trake" ? "TRAKE" : "T-KIS") as QueryType,
    text: "",
    eventCount,
  });
  const pack = [q("query-1-kis", "kis"), q("query-2-kis", "kis"), q("query-3-qa", "qa")];
  const good = (over: Partial<SubmissionRow>): SubmissionRow => ({
    id: "r", questionId: "query-1-kis", videoId: "L01_V028", frames: [3450], answer: "", ...over,
  });

  it("writes one CSV per question, including the unanswered ones", () => {
    // The spec asks for a file per query and shows all of them in the sample
    // tree; a missing file is a different thing to the grader than an empty one.
    const rows = [good({ id: "a" })];
    const text = new TextDecoder().decode(buildSubmissionZipBytes(pack, rows));
    for (const question of pack) expect(text).toContain(`submission/${question.id}.csv`);
    expect(auditSubmission(pack, rows).unanswered).toEqual(["query-2-kis", "query-3-qa"]);
  });

  it("refuses to build a zip when any row would be rejected", () => {
    const cases: [string, SubmissionRow][] = [
      [".mp4", good({ videoId: "L01_V028.mp4" })],
      ["thiếu video", good({ videoId: "  " })],
      ["thiếu frame", good({ frames: [] })],
    ];
    for (const [label, row] of cases) {
      expect(() => buildSubmissionZip(pack, [row]), label).toThrow(SubmissionFormatError);
    }
    // QA rules apply to the QA question only.
    expect(() =>
      buildSubmissionZip(pack, [good({ questionId: "query-3-qa", answer: "x".repeat(101) })]),
    ).toThrow(SubmissionFormatError);
  });

  it("blocks a TRAKE row with the wrong count or order", () => {
    const trakePack = [q("query-4-trake", "trake", 3)];
    const row = (frames: number[]) => good({ questionId: "query-4-trake", frames });
    expect(() => buildSubmissionZip(trakePack, [row([10, 20])])).toThrow(SubmissionFormatError);
    expect(() => buildSubmissionZip(trakePack, [row([10, 9, 30])])).toThrow(SubmissionFormatError);
    expect(() => buildSubmissionZip(trakePack, [row([10, 20, 30])])).not.toThrow();
  });

  it("names every offending question in the refusal", () => {
    try {
      buildSubmissionZip(pack, [good({ id: "a", videoId: "L01_V028.mp4" })]);
      throw new Error("expected a refusal");
    } catch (error) {
      expect(error).toBeInstanceOf(SubmissionFormatError);
      const problems = (error as SubmissionFormatError).problems;
      expect(problems[0].questionId).toBe("query-1-kis");
      expect((error as Error).message).toContain(".mp4");
    }
  });

  it("lets a duplicate through — the spec does not forbid predicting twice", () => {
    const rows = [good({ id: "a" }), good({ id: "b" })];
    const audit = auditSubmission(pack, rows);
    expect(audit.errors).toHaveLength(0);
    expect(audit.warnings.map((w) => w.message).join(" ")).toContain("duplicates");
    expect(() => buildSubmissionZip(pack, rows)).not.toThrow();
  });
});
