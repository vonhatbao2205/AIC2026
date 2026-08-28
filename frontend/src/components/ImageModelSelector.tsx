import type { ImageEmbeddingModel, RetrievalDatabase } from "../api/types";
import { imageModelsForSearch, toggleImageModel } from "../lib/imageModels";

interface Props {
  retrievalDatabase: RetrievalDatabase;
  value: ImageEmbeddingModel[];
  onChange: (models: ImageEmbeddingModel[]) => void;
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

/** InfoShot++ image-index picker. Selecting both asks the backend to RRF-fuse them. */
export function ImageModelSelector({ retrievalDatabase, value, onChange }: Props) {
  if (retrievalDatabase !== "infoshotpp") return null;

  const selected = imageModelsForSearch(retrievalDatabase, value);
  const usingRrf = selected.length === OPTIONS.length;

  return (
    <fieldset
      className="image-model-selector"
      data-testid="image-model-selector"
      title={usingRrf ? "Hai model được tìm song song và fuse bằng RRF." : "Chọn image embedding cho lần search tiếp theo."}
    >
      <legend>Image embedding</legend>
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
      {usingRrf && (
        <span className="image-model-fusion" data-testid="image-model-fusion">
          RRF
        </span>
      )}
    </fieldset>
  );
}
