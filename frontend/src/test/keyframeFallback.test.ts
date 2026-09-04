import { afterEach, describe, expect, it } from "vitest";
import { installKeyframeFallback } from "../lib/keyframeFallback";

const R2 = "https://keyframe.baoencoder.site";
const HF = "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve";
const PATH = "/Keyframes/Keyframes_L26/L26_V001/001.jpg";

let uninstall = () => {};

afterEach(() => {
  uninstall();
  uninstall = () => {};
  document.body.innerHTML = "";
});

function image(src: string): HTMLImageElement {
  const img = document.createElement("img");
  img.src = src;
  document.body.appendChild(img);
  return img;
}

function fail(img: HTMLImageElement): void {
  img.dispatchEvent(new Event("error"));
}

describe("keyframe origin fallback", () => {
  it("retries a failed R2 keyframe from Hugging Face", () => {
    uninstall = installKeyframeFallback(R2, HF);
    const img = image(R2 + PATH);

    fail(img);

    expect(img.getAttribute("src")).toBe(HF + PATH);
  });

  it("covers keyframes added after the listener is installed", () => {
    uninstall = installKeyframeFallback(R2, HF);
    const img = image(`${R2}/Keyframes/Keyframes_L30/L30_V009/042.jpg`);

    fail(img);

    expect(img.getAttribute("src")).toBe(
      `${HF}/Keyframes/Keyframes_L30/L30_V009/042.jpg`,
    );
  });

  it("retries each image only once", () => {
    uninstall = installKeyframeFallback(R2, HF);
    const img = image(R2 + PATH);

    fail(img);
    fail(img);

    expect(img.getAttribute("src")).toBe(HF + PATH);
  });

  it("does not rewrite images from another origin", () => {
    uninstall = installKeyframeFallback(R2, HF);
    const img = image("https://other.example/Keyframes/L21/001.jpg");

    fail(img);

    expect(img.getAttribute("src")).toBe("https://other.example/Keyframes/L21/001.jpg");
  });

  it("does nothing without a fallback", () => {
    uninstall = installKeyframeFallback(R2, "");
    const img = image(R2 + PATH);

    fail(img);

    expect(img.getAttribute("src")).toBe(R2 + PATH);
  });

  it("stops listening once uninstalled", () => {
    const stop = installKeyframeFallback(R2, HF);
    stop();
    const img = image(R2 + PATH);

    fail(img);

    expect(img.getAttribute("src")).toBe(R2 + PATH);
  });
});
