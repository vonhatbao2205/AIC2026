import type { ImageEmbeddingModel, RetrievalDatabase } from "../api/types";
import { imageModelsForSearch, toggleImageModel } from "../lib/imageModels";

interface Props {
  retrievalDatabase: RetrievalDatabase;
  value: ImageEmbeddingModel[];
  onChange: (models: ImageEmbeddingModel[]) => void;
  /** TARA is a temporal clip model, so it is controlled through the channel
   * override payload rather than `image_models`. Keeping it in this visual
   * model picker makes that implementation detail invisible to the operator. */
  tara?: {
    available: boolean;
    enabled: boolean;
    onChange: (enabled: boolean) => void;
  };
  /** The VISUAL channel (`image_pe`), which runs every keyframe model. Unticking
   * the last keyframe model switches it off, which leaves TARA searching alone;
   * that is only allowed while TARA is on, so a search always has a visual model. */
  keyframes?: {
    enabled: boolean;
    onChange: (enabled: boolean) => void;
  };
}

const OPTIONS: { id: ImageEmbeddingModel; label: string; title: string }[] = [
  {
    id: "pe",
    label: "PE Core",
    title: "Search InfoShot++ using PE Core embeddings.",
  },
  {
    id: "qwen3_vl",
    label: "Qwen3-VL Embedding",
    title: "Search InfoShot++ using Qwen3-VL Embedding 8B.",
  },
];

/** InfoShot++ visual-model picker. PE/Qwen rank keyframes; TARA ranks clips. */
export function ImageModelSelector({ retrievalDatabase, value, onChange, tara, keyframes }: Props) {
  if (retrievalDatabase !== "infoshotpp") return null;

  const usingTara = Boolean(tara?.available && tara.enabled);
  const keyframesOn = keyframes?.enabled ?? true;
  // With the VISUAL channel off no keyframe model runs, whatever `value` holds;
  // `value` is kept so ticking a model again restores the operator's choice.
  const selected = keyframesOn ? imageModelsForSearch(retrievalDatabase, value) : [];
  const usingImageRrf = selected.length === OPTIONS.length;

  function toggle(model: ImageEmbeddingModel) {
    if (!keyframesOn) {
      onChange([model]);
      keyframes?.onChange(true);
    } else if (selected.length === 1 && selected[0] === model) {
      keyframes?.onChange(false);
    } else {
      onChange(toggleImageModel(selected, model));
    }
  }

  return (
    <fieldset
      className="image-model-selector"
      data-testid="image-model-selector"
      title="PE/Qwen retrieve keyframes; TARA retrieves 8/24/72-second clips (4/8/16 s on traffic cameras). Rankings are fused using RRF."
    >
      <legend>Visual models</legend>
      {OPTIONS.map((option) => {
        const checked = selected.includes(option.id);
        // The last keyframe model can go only when TARA is left to search.
        const soleSelection = checked && selected.length === 1 && !(keyframes && usingTara);
        return (
          <label
            className="check-toggle"
            key={option.id}
            title={soleSelection
              ? (tara?.available
                ? "Tick TARA clips to search without a keyframe model."
                : "Select at least one image embedding model to search.")
              : option.title}
          >
            <input
              type="checkbox"
              checked={checked}
              disabled={soleSelection}
              data-testid={`image-model-${option.id}`}
              onChange={() => toggle(option.id)}
            />
            <span>{option.label}</span>
          </label>
        );
      })}
      {usingImageRrf && (
        <span className="image-model-fusion" data-testid="image-model-fusion">
          keyframe RRF
        </span>
      )}
      {tara && (
        <label
          className="check-toggle tara-model-option"
          title={!tara.available
            ? "TARA requires an available Colab text encoder and Milvus collection."
            : usingTara && !keyframesOn
              ? "TARA is the only visual model; tick PE Core or Qwen3-VL before unticking it."
              : "TARA searches motion and events in 8/24/72-second clips (4/8/16 s on traffic cameras); results are fused at video level."}
        >
          <input
            type="checkbox"
            checked={usingTara}
            disabled={!tara.available || (usingTara && !keyframesOn)}
            data-testid="model-tara"
            onChange={(event) => tara.onChange(event.target.checked)}
          />
          <span>TARA clips{tara.available ? "" : " · offline"}</span>
        </label>
      )}
      {usingTara && (
        <span className="image-model-fusion tara-fusion" data-testid="tara-video-fusion">
          {keyframesOn ? "video RRF" : "TARA only"}
        </span>
      )}
    </fieldset>
  );
}
