import { describe, expect, it } from "vitest";
import {
  parseQuestionEntry,
  parseQuestionPack,
  trakeEventCount,
  kindForQueryType,
} from "../lib/questions";
import {
  csvField,
  findDuplicate,
  questionCsv,
  rowKey,
  rowToCsvLine,
  validateRows,
  type SubmissionRow,
} from "../lib/submission";
import { buildZipBytes, crc32 } from "../lib/zip";
import {
  canonicalQuestions,
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
    expect(problems.some((p) => p.message.includes("cần đúng 4 frame"))).toBe(true);
  });

  it("flags TRAKE frames that do not increase", () => {
    const problems = validateRows([row({ frames: [10, 9, 30, 40] })], trake);
    expect(problems.some((p) => p.message.includes("tăng dần"))).toBe(true);
  });

  it("flags an over-long or empty QA answer", () => {
    expect(validateRows([row({ answer: "" })], qa).some((p) => p.message.includes("thiếu answer"))).toBe(true);
    expect(
      validateRows([row({ answer: "x".repeat(101) })], qa).some((p) => p.message.includes("> 100")),
    ).toBe(true);
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
    expect(describeSupabaseError(null)).toBe("Lỗi không xác định");
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
    {
      id: "query-p1-4-trake", order: over.order ?? 4, kind: "trake" as const,
      queryType: "TRAKE" as const, text: over.text ?? "E1: x\nE2: y", eventCount: 2,
    },
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

  it("summarises a pack for the publish confirmation", () => {
    expect(packSummary(pack())).toEqual({ total: 2, byKind: { kis: 1, qa: 0, trake: 1 } });
  });
});
