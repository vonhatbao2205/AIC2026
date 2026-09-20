/**
 * The setup path of the packaged app: a build downloaded with no credentials has
 * to explain itself and let the operator import the team's `.env` from the UI.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { getDisplayName, setDisplayName, subscribeDisplayName } from "../lib/supabase";

const GROUPS = [
  {
    name: "Elastic Cloud",
    summary: "OCR, speech, audio and timeline lookups.",
    keys: [
      { key: "ELASTIC_ENDPOINT", label: "Cluster endpoint URL", secret: false, required: true, set: true, preview: "https://cluster.es.io", from_process_env: false },
      { key: "ELASTIC_API_KEY", label: "API key", secret: true, required: true, set: false, preview: "", from_process_env: false },
    ],
  },
  {
    name: "Runtime",
    summary: "Local behaviour of this instance.",
    keys: [
      { key: "AIC26_MOCK_MODE", label: "Run on fixtures", secret: false, required: false, set: true, preview: "false", from_process_env: true },
    ],
  },
];

const UNCONFIGURED = {
  configured: false,
  mock_mode: false,
  missing_required: ["ELASTIC_API_KEY", "MILVUS_TOKEN"],
  env_path: "/config/.env",
  env_exists: false,
  groups: GROUPS,
};

const CONFIGURED = { ...UNCONFIGURED, configured: true, missing_required: [], env_exists: true };

const HEALTH = {
  ok: true,
  mode: "live",
  services: {},
  capabilities: {},
  warnings: [],
};

let configStatus: any = UNCONFIGURED;
let imports: { replace: string; body: string }[] = [];

/** jsdom's File has no `.text()`, so read the upload the long way. */
function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function mockFetch() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);

    if (path.endsWith("/api/config/import")) {
      const form = init?.body as FormData;
      const file = form.get("file") as File;
      imports.push({ replace: String(form.get("replace")), body: await readFile(file) });
      configStatus = CONFIGURED;
      return json({
        env_path: "/config/.env",
        applied: ["ELASTIC_API_KEY", "MILVUS_TOKEN"],
        unknown: [],
        rejected: [],
        ignored_blank: ["DRES_PASSWORD"],
        replaced: form.get("replace") === "true",
        status: CONFIGURED,
      });
    }
    if (path.endsWith("/api/config")) return json(configStatus);
    if (path.endsWith("/api/health")) return json(HEALTH);
    if (path.includes("/api/submit/history")) return json({ history: [] });
    if (path.endsWith("/api/dres/status")) return json({ configured: false, ok: false, mode: "disabled" });
    return json({}, 404);
  });
}

beforeEach(() => {
  globalThis.localStorage?.clear?.();
  configStatus = UNCONFIGURED;
  imports = [];
  vi.stubGlobal("fetch", mockFetch());
});
afterEach(() => vi.unstubAllGlobals());

describe("configuration", () => {
  it("opens itself when the backend has no credentials, and names what is missing", async () => {
    render(<App />);

    const modal = await screen.findByTestId("settings-modal");
    expect(await screen.findByTestId("settings-needs-setup")).toBeInTheDocument();
    expect(modal).toHaveTextContent("ELASTIC_API_KEY, MILVUS_TOKEN");
  });

  it("stays out of the way once the instance is configured", async () => {
    configStatus = CONFIGURED;
    render(<App />);

    await screen.findByTestId("settings-btn");
    expect(screen.queryByTestId("settings-modal")).toBeNull();
  });

  it("imports an uploaded .env and reports what the backend did with it", async () => {
    render(<App />);
    await screen.findByTestId("settings-modal");

    const file = new File(["ELASTIC_API_KEY=k\nMILVUS_TOKEN=t\n"], ".env", { type: "text/plain" });
    fireEvent.change(screen.getByTestId("settings-file-input"), { target: { files: [file] } });
    await screen.findByText(/ready to import/);

    fireEvent.click(screen.getByTestId("settings-import"));

    const result = await screen.findByTestId("settings-result");
    expect(result).toHaveTextContent("Applied 2 variables");
    // A half-filled template must not silently wipe a working secret.
    expect(result).toHaveTextContent("Skipped 1 blank values");
    expect(imports).toEqual([{ replace: "false", body: "ELASTIC_API_KEY=k\nMILVUS_TOKEN=t\n" }]);
    // The setup warning clears because the backend now reports it configured.
    await waitFor(() => expect(screen.queryByTestId("settings-needs-setup")).toBeNull());
  });

  it("sends the replace flag only when the operator asks for it", async () => {
    render(<App />);
    await screen.findByTestId("settings-modal");
    const user = userEvent.setup();

    await user.click(screen.getByLabelText(/Replace entire configuration/));
    const file = new File(["ELASTIC_API_KEY=k\n"], ".env", { type: "text/plain" });
    fireEvent.change(screen.getByTestId("settings-file-input"), { target: { files: [file] } });
    fireEvent.click(screen.getByTestId("settings-import"));

    await screen.findByTestId("settings-result");
    expect(imports[0].replace).toBe("true");
  });

  it("shows secrets only as the backend masked them, and flags container-set values", async () => {
    configStatus = {
      ...CONFIGURED,
      groups: [
        {
          ...GROUPS[0],
          keys: [
            { ...GROUPS[0].keys[1], set: true, preview: "••••••••wxyz" },
          ],
        },
        GROUPS[1],
      ],
    };
    render(<App />);
    fireEvent.click(await screen.findByTestId("settings-btn"));

    const modal = await screen.findByTestId("settings-modal");
    expect(modal).toHaveTextContent("••••••••wxyz");
    // The env tag tells the operator an import cannot change this one.
    expect(modal).toHaveTextContent("env");
  });

  it("is reachable from the console after setup", async () => {
    configStatus = CONFIGURED;
    render(<App />);

    fireEvent.click(await screen.findByTestId("settings-btn"));
    expect(await screen.findByTestId("settings-modal")).toBeInTheDocument();
  });
});

/**
 * One packaged image serves the whole team, so `VITE_SUBMISSION_USER` cannot
 * carry a per-person name — every operator would submit as the same string.
 * The name is therefore set here and stored per machine.
 */
describe("operator name", () => {
  it("stores the typed name so submitted rows are attributed", async () => {
    configStatus = CONFIGURED;
    render(<App />);
    fireEvent.click(await screen.findByTestId("settings-btn"));
    await screen.findByTestId("settings-identity");
    const user = userEvent.setup();

    await user.type(screen.getByTestId("settings-name-input"), "Bao");
    await user.click(screen.getByTestId("settings-name-save"));

    expect(await screen.findByTestId("settings-name-saved")).toBeInTheDocument();
    expect(getDisplayName()).toBe("Bao");
  });

  it("keeps the name across a reopen of the modal", async () => {
    configStatus = CONFIGURED;
    setDisplayName("Huy");
    render(<App />);

    fireEvent.click(await screen.findByTestId("settings-btn"));
    await screen.findByTestId("settings-identity");
    expect(screen.getByTestId("settings-name-input")).toHaveValue("Huy");
  });

  it("trims surrounding whitespace rather than storing it", async () => {
    configStatus = CONFIGURED;
    render(<App />);
    fireEvent.click(await screen.findByTestId("settings-btn"));
    await screen.findByTestId("settings-identity");
    const user = userEvent.setup();

    await user.type(screen.getByTestId("settings-name-input"), "  Minh  ");
    await user.click(screen.getByTestId("settings-name-save"));

    expect(getDisplayName()).toBe("Minh");
  });

  it("notifies subscribers, so open views relabel without a reload", async () => {
    const seen: string[] = [];
    const unsubscribe = subscribeDisplayName(() => seen.push(getDisplayName()));
    setDisplayName("Lan");
    unsubscribe();
    setDisplayName("Ignored");

    expect(seen).toEqual(["Lan"]);
  });
});
