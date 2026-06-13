import { useState } from "react";
import FullConsole from "./FullConsole";
import SimpleSearch from "./SimpleSearch";

type View = "console" | "simple";

export default function App() {
  const [view, setView] = useState<View>(() => {
    const saved = typeof localStorage !== "undefined" ? localStorage.getItem("aic26_view") : null;
    return saved === "simple" ? "simple" : "console";
  });

  function go(v: View) {
    setView(v);
    try {
      localStorage.setItem("aic26_view", v);
    } catch {
      /* ignore */
    }
  }

  return view === "simple" ? (
    <SimpleSearch onFullMode={() => go("console")} />
  ) : (
    <FullConsole onSimpleMode={() => go("simple")} />
  );
}
