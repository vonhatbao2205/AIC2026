import { describe, expect, it } from "vitest";
import type { DresEvaluation } from "../api/types";
import { autoEvaluationId, matchesQueryType } from "../lib/dres";

function evaluation(partial: Partial<DresEvaluation> & { id: string; name: string }): DresEvaluation {
  return {
    status: "ACTIVE",
    type: "SYNCHRONOUS",
    teams: [],
    task_templates: [],
    current_task: null,
    state: null,
    ...partial,
  } as DresEvaluation;
}

const TKIS = evaluation({
  id: "e-tkis",
  name: "tkis",
  task_templates: [{ name: "tkis-00", taskGroup: "T-KIS Group", taskType: "Textual KIS", duration: null }],
  current_task: { name: "tkis-00", taskGroup: "T-KIS Group", taskType: "Textual KIS", duration: null },
});
const QA = evaluation({
  id: "e-qa",
  name: "qa",
  task_templates: [{ name: "qa-00", taskGroup: "QA Group", taskType: "Question Answering", duration: null }],
  current_task: { name: "qa-00", taskGroup: "QA Group", taskType: "Question Answering", duration: null },
});
const VKIS = evaluation({
  id: "e-vkis",
  name: "vkis",
  task_templates: [{ name: "vkis-00", taskGroup: "V-KIS Group", taskType: "Visual KIS", duration: null }],
});

describe("DRES evaluation routing", () => {
  it("matches a run by its name and task templates", () => {
    expect(matchesQueryType(TKIS, "T-KIS")).toBe(true);
    expect(matchesQueryType(TKIS, "QA")).toBe(false);
    expect(matchesQueryType(QA, "QA")).toBe(true);
    expect(matchesQueryType(VKIS, "V-KIS")).toBe(true);
    // "tkis" must not be read as a visual run, nor "vkis" as a textual one.
    expect(matchesQueryType(VKIS, "T-KIS")).toBe(false);
    expect(matchesQueryType(TKIS, "V-KIS")).toBe(false);
  });

  it("picks the run of the current query type", () => {
    const all = [TKIS, QA, VKIS];
    expect(autoEvaluationId(all, "T-KIS")).toBe("e-tkis");
    expect(autoEvaluationId(all, "QA")).toBe("e-qa");
    expect(autoEvaluationId(all, "V-KIS")).toBe("e-vkis");
  });

  it("prefers the run that actually has a task open", () => {
    const stale = evaluation({ id: "e-old", name: "tkis warmup", task_templates: TKIS.task_templates });
    expect(autoEvaluationId([stale, TKIS], "T-KIS")).toBe("e-tkis");
  });

  it("falls back to the only active run when nothing matches by name", () => {
    const only = evaluation({ id: "e-1", name: "finals", current_task: TKIS.current_task });
    expect(autoEvaluationId([only], "TRAKE")).toBe("e-1");
  });

  it("returns null rather than guessing between two unrelated runs", () => {
    const a = evaluation({ id: "a", name: "round one" });
    const b = evaluation({ id: "b", name: "round two" });
    expect(autoEvaluationId([a, b], "TRAKE")).toBeNull();
  });

  it("ignores runs that are not active when an active one exists", () => {
    const ended = evaluation({ id: "e-done", name: "qa", status: "TERMINATED" });
    expect(autoEvaluationId([ended, QA], "QA")).toBe("e-qa");
  });
});
