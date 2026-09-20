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
}

const OPTIONS: { id: ImageEmbeddingModel; label: string; title: string }[] = [
  {
    id: "pe",
    label: "PE Core",
    title: "Tìm bằng embedding PE Core trên InfoShot++.",
  },
  {
    id: "qwen3_vl",
    label: "Qwen3-VL Embedding",
    title: "Tìm bằng Qwen3-VL Embedding 8B trên InfoShot++.",
  },
];

/** InfoShot++ visual-model picker. PE/Qwen rank keyframes; TARA ranks clips. */
export function ImageModelSelector({ retrievalDatabase, value, onChange, tara }: Props) {
  if (retrievalDatabase !== "infoshotpp") return null;

  const selected = imageModelsForSearch(retrievalDatabase, value);
  const usingImageRrf = selected.length === OPTIONS.length;
  const usingTara = Boolean(tara?.available && tara.enabled);

  return (
    <fieldset
      className="image-model-selector"
      data-testid="image-model-selector"
      title="PE/Qwen tìm keyframe; TARA tìm clip 8/24/72 giây. Các ranking được fuse bằng RRF."
    >
      <legend>Visual models</legend>
      {OPTIONS.map((option) => {
        const checked = selected.includes(option.id);
        const soleSelection = checked && selected.length === 1;
        return (
          <label
            className="check-toggle"
            key={option.id}
            title={soleSelection ? "Phải giữ ít nhất một image embedding để search." : option.title}
          >
            <input
              type="checkbox"
              checked={checked}
              disabled={soleSelection}
              data-testid={`image-model-${option.id}`}
              onChange={() => onChange(toggleImageModel(selected, option.id))}
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
          title={tara.available
            ? "TARA tìm theo chuyển động và diễn biến trong clip 8/24/72 giây; kết quả được fuse ở mức video."
            : "TARA cần text encoder Colab và collection Milvus sẵn sàng."}
        >
          <input
            type="checkbox"
            checked={usingTara}
            disabled={!tara.available}
            data-testid="model-tara"
            onChange={(event) => tara.onChange(event.target.checked)}
          />
          <span>TARA clips{tara.available ? "" : " · offline"}</span>
        </label>
      )}
      {usingTara && (
        <span className="image-model-fusion tara-fusion" data-testid="tara-video-fusion">
          video RRF
        </span>
      )}
    </fieldset>
  );
}
