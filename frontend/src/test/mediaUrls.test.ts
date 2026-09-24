import { describe, expect, it } from "vitest";
import { groupFromVideoId, keyframeUrl, keyframeUrlFromSubmitId, videoUrl } from "../lib/media";

const BASE = "https://media.test/";

describe("media URL helpers", () => {
  it("keeps the underscore convention of the L/K/M folders", () => {
    expect(groupFromVideoId("L21_V001")).toBe("L21");
    expect(keyframeUrl(BASE, "M01_V001", 7)).toBe("https://media.test/Keyframes/Keyframes_M01/M01_V001/007.jpg");
  });

  it("resolves the hyphenated batch-2 camera and cycling ids to their folder", () => {
    expect(groupFromVideoId("N001-V001")).toBe("N001");
    expect(keyframeUrlFromSubmitId(BASE, "N001/N001-V001/002")).toBe(
      "https://media.test/Keyframes/Keyframes_N001/N001-V001/002.jpg",
    );
    expect(videoUrl(BASE, "S01-V007")).toBe("https://media.test/Videos/Videos_S01/S01-V007.mp4");
    expect(videoUrl(BASE, "N001-V001")).toBe("https://media.test/Videos_Web/Videos_N001/N001-V001.mp4");
  });
});
