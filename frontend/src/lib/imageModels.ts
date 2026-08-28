import type { ImageEmbeddingModel, RetrievalDatabase } from "../api/types";

export const DEFAULT_IMAGE_MODELS: readonly ImageEmbeddingModel[] = ["pe"];

const MODEL_ORDER: ImageEmbeddingModel[] = ["pe", "qwen3_vl"];

/**
 * Keep the request deterministic and valid even if stale UI state is restored.
 * Qwen3-VL is an InfoShot++ index; every BTC request deliberately stays PE-only.
 */
export function imageModelsForSearch(
  retrievalDatabase: RetrievalDatabase,
  selected: readonly ImageEmbeddingModel[],
): ImageEmbeddingModel[] {
  if (retrievalDatabase !== "infoshotpp") return [...DEFAULT_IMAGE_MODELS];
  const held = new Set(selected);
  const normalized = MODEL_ORDER.filter((model) => held.has(model));
  return normalized.length ? normalized : [...DEFAULT_IMAGE_MODELS];
}

export function toggleImageModel(
  selected: readonly ImageEmbeddingModel[],
  model: ImageEmbeddingModel,
): ImageEmbeddingModel[] {
  const current = imageModelsForSearch("infoshotpp", selected);
  if (current.includes(model)) {
    // At least one image index must answer every search.
    return current.length === 1 ? current : current.filter((item) => item !== model);
  }
  return MODEL_ORDER.filter((item) => item === model || current.includes(item));
}
