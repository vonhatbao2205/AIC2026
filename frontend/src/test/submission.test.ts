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
