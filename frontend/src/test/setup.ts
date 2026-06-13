import "@testing-library/jest-dom/vitest";

// jsdom lacks these; stub for components that touch them.
if (!("clipboard" in navigator)) {
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText: () => Promise.resolve() },
    configurable: true,
  });
}
