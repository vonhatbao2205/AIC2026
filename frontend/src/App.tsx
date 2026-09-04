import { useEffect, useState } from "react";
import { api } from "./api/client";
import { installKeyframeFallback } from "./lib/keyframeFallback";
import Workspace from "./Workspace";
import SimpleSearch from "./SimpleSearch";
import { SettingsModal } from "./components/SettingsModal";
import type { RetrievalDatabase } from "./api/types";

type View = "console" | "simple";

export default function App() {
  const [view, setView] = useState<View>(() => {
    const saved = typeof localStorage !== "undefined" ? localStorage.getItem("aic26_view") : null;
    return saved === "simple" ? "simple" : "console";
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [retrievalDatabase, setRetrievalDatabase] = useState<RetrievalDatabase>(() => {
    const saved = typeof localStorage !== "undefined"
      ? localStorage.getItem("aic26_retrieval_database")
      : null;
    return saved === "infoshotpp" ? "infoshotpp" : "btc";
  });
  // Bumped after a config import so the console re-reads /api/health against the
  // services the backend has just rebuilt.
  const [configVersion, setConfigVersion] = useState(0);

  // One delegated listener covers every keyframe rendered by either app view.
  // Reinstall it when the selected dataset or imported configuration changes.
  useEffect(() => {
    let disposed = false;
    let uninstall = () => {};

    api
      .health(retrievalDatabase)
      .then((health) => {
        if (disposed) return;
        uninstall = installKeyframeFallback(
          health.media?.keyframe_base_url,
          health.media?.keyframe_fallback_base_url,
        );
      })
      .catch(() => {
        /* backend unavailable: images remain on the URLs already returned */
      });

    return () => {
      disposed = true;
      uninstall();
    };
  }, [retrievalDatabase, configVersion]);

  useEffect(() => {
    // A freshly downloaded build carries no credentials. Open the setup screen
    // rather than letting the operator's first search fail unexplained.
    api
      .config()
      .then((cfg) => {
        if (!cfg.configured) setSettingsOpen(true);
      })
      .catch(() => {
        /* backend not up yet — the console shows its own connection error */
      });
  }, []);

  function go(v: View) {
    setView(v);
    try {
      localStorage.setItem("aic26_view", v);
    } catch {
      /* ignore */
    }
  }

  function chooseDatabase(database: RetrievalDatabase) {
    setRetrievalDatabase(database);
    try {
      localStorage.setItem("aic26_retrieval_database", database);
    } catch {
      /* ignore */
    }
  }

  return (
    <>
      {view === "simple" ? (
        <SimpleSearch
          retrievalDatabase={retrievalDatabase}
          onRetrievalDatabase={chooseDatabase}
          onFullMode={() => go("console")}
          onShowSettings={() => setSettingsOpen(true)}
        />
      ) : (
        <Workspace
          onSimpleMode={() => go("simple")}
          onShowSettings={() => setSettingsOpen(true)}
          configVersion={configVersion}
          retrievalDatabase={retrievalDatabase}
          onRetrievalDatabase={chooseDatabase}
        />
      )}
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onApplied={() => setConfigVersion((v) => v + 1)}
      />
    </>
  );
}
