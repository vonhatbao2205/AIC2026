# Object Detection Pipeline — Zero-Shot, No Fine-Tune (mid-2026)

> **Phạm vi:** CHỈ task Object Detection trên keyframe video tin tức tiếng Việt.
> **Input:** ảnh keyframe (.jpg). **Output:** `{label, bbox, conf, color, grid_cell, crop_embedding}` cho mỗi object.
> **Ràng buộc:** open-weight, serving local, **zero-shot — KHÔNG fine-tune, KHÔNG gán nhãn tay**.

---

## 1. Vì sao zero-shot open-vocab (không phải YOLOv8/COCO)

Domain tin tức Việt có entity **không nằm trong COCO/OpenImages**: logo đài (VTV/HTV), lower-third graphics, biển hiệu, đồng phục công an/bộ đội, MC. Detector closed-set (YOLOv8/v12, Co-DETR — thứ đa số đội AIC 2025 dùng) chỉ biết ~80-600 lớp cố định → **miss hết entity đặc thù**. Vì không thể fine-tune (không có nhãn), bắt buộc dùng **open-vocabulary text-prompted detection** + **VLM grounding**.

---

## 2. Kiến trúc 3 model (ensemble zero-shot)

```
keyframe ──┬─→ [A] YOLOE (prompt-free + text)      ─ fast, recall rộng ──┐
           ├─→ [B] MM-Grounding-DINO (text-prompt) ─ accurate, concept ──┼─→ WBF fusion ─→ detections
           └─→ [C] Qwen3-VL grounding (on-demand)  ─ dense/đếm/mơ hồ ─────┘
                                                                            │
                              post-process: WBF → color → 7×7 grid → crop-embed → JSONB
```

| Model | Vai trò | Khi nào chạy | License |
|---|---|---|---|
| **[A] YOLOE-26-L** | Bulk tagging real-time, **NMS-free**. Prompt-free (`yoloe-26l-seg-pf.pt`) tự sinh tag + text-prompt (`yoloe-26l-seg.pt`) theo vocab §3. Có sẵn instance mask. 36.8 LVIS mAP, 161 FPS T4. Strict upgrade từ YOLOE (+1.6 mAP, cùng tốc độ) | TẤT CẢ keyframe (offline) | open-weight (Ultralytics) |
| **[B] MM-Grounding-DINO** | Concept-specific chính xác hơn YOLOE, recall cao cho rare class | TẤT CẢ keyframe (offline) | Apache 2.0 |
| **[C] Qwen3-VL grounding** | Dense scene (đám đông), đếm, entity mơ hồ mà box-detector fail | Chỉ keyframe khó / khi cần đếm | open-weight |

> Không dùng DINO-X / Grounding-DINO 1.6 Pro (mạnh hơn nhưng **API-only** → vi phạm serving local).

**Tại sao 2 detector + 1 VLM:** YOLOE nhanh nhưng recall rare-class kém hơn; MM-GDINO chính xác hơn nhưng chậm; chạy cả 2 rồi WBF cho recall + precision tốt nhất. VLM chỉ gọi khi 2 detector bất đồng hoặc cần đếm dày.

---

## 3. Concept vocabulary (quan trọng nhất — quyết định chất lượng open-vocab)

Open-vocab detector chỉ tốt khi **prompt list được thiết kế kỹ**. **Dataset KHÔNG đồng nhất là tin tức** → dùng **vocab theo thể loại (genre-conditional)**: mỗi keyframe chỉ được prompt bằng concept hợp với chương trình của nó (ít nhiễu, nhanh hơn, recall cao hơn). Kiến trúc = `BASE_VOCAB` (áp mọi keyframe) + `GENRE_VOCAB[genre]`.

**Map batch → genre** (theo ghi chép xem dữ liệu):

| Batch | Thể loại | Genre key |
|---|---|---|
| L21, L22, **K01–K20** | Thời sự 60 giây (HTV7/HTV9) | `news` |
| L23 | Đua xe đạp | `cycling` |
| L24 | Múa lân | `lion_dance` |
| L25 | Ôn thi THPTQG | `exam` |
| L26 (a–e) | Nấu ăn | `cooking` |
| L27 | Văn hóa Việt Nam | `culture` |
| L28, L29 | Vẻ đẹp/văn hóa lưu vực Mekong | `mekong` |
| L30 | Short năng lượng tích cực (Tuổi trẻ TV) | `positive` |

→ Phần lớn data là `news` (toàn bộ K + L21/L22), nhưng `cooking` (5 batch L26) và các genre còn lại cần vocab riêng (food/utensil/stove vs reporter/fire/flood). Vocab cụ thể từng genre xem `BASE_VOCAB` + `GENRE_VOCAB` trong notebook (`OD_Kaggle_Notebook.ipynb`, ô Config). `genre` được lưu vào output → faceted filter theo chương trình.

**Kỹ thuật tăng chất lượng prompt:**
- **Synonym expansion:** mỗi concept thêm 2-3 cách diễn đạt (`"motorbike" / "motorcycle" / "scooter"`) → search song song, gộp box.
- **Hierarchical prompting:** chạy generic trước (`"vehicle"`, `"person"`) rồi fine concept (`"ambulance"`) → tăng recall.
- **Per-query dynamic vocab:** lúc retrieval, nếu query nhắc concept lạ → thêm vào prompt MM-GDINO chạy lại trên candidate (open-vocab cho phép thêm runtime, không cần re-index).
- **Logo/graphics:** open-vocab detect logo kém → để **OCR cứu** (logo đài thường có chữ); chỉ tag bbox vùng "tv logo" làm gợi ý.

---

## 4. Ensemble fusion — Weighted Box Fusion (WBF)

Gộp output [A]+[B] (và [C] nếu có). **WBF tốt hơn NMS** cho ensemble vì nó *trung bình* box thay vì loại bỏ:

```
Input: boxes từ YOLOE + MM-GDINO (+ Qwen3-VL), mỗi box {label, bbox, conf, model_weight}
- Khớp box cùng label, IoU > 0.55
- Box hợp nhất = weighted average tọa độ theo conf×model_weight
- conf hợp nhất = max(conf) hoặc weighted (tùy calibrate)
model_weight gợi ý: MM-GDINO 1.0, YOLOE 0.8, Qwen3-VL 1.2 (cho dense)
```

Lý do: 2 detector cùng vote 1 vùng → box chính xác hơn + conf cao hơn; box chỉ 1 model thấy → giữ với conf giảm (recall mà không quá nhiễu).

---

## 5. Hậu xử lý — attributes (rẻ, ROI cao)

Học từ CLIPAR/TEMPO (AIC 2025) — biến detection thành feature giàu cho query compositional:

**5.1 Spatial — lưới 7×7:**
- Map tâm mỗi bbox vào ô lưới 7×7 (`grid_cell = (row, col)`).
- → query "object ở góc trái trên", "người bên phải".

**5.2 Color — 16-color palette:**
- Crop bbox → quantize về palette 16 màu → dominant color của object.
- → query "biển hiệu màu đỏ", "áo xanh".

**5.3 Count:**
- Đếm instance mỗi label trong frame → `count[label]`.
- Dense scene (>20 người): box-detector unreliable → gọi **Qwen3-VL** đếm hoặc dùng crowd-density estimate. Lưu `count_method` để biết độ tin.

**5.4 Instance embedding (ATS trick):**
- Crop bbox conf>0.5 → encode bằng **PE-Core** (hoặc DINOv2) → vector.
- → instance retrieval "tìm đúng chiếc xe/logo này", không chỉ class-level.

---

## 6. Storage schema

```json
{
  "keyframe_id": ["L01_V001", 1234],
  "detections": [
    {
      "label": "ambulance",
      "label_group": "vehicles",
      "bbox": [x1,y1,x2,y2],
      "conf": 0.87,
      "source": ["mm_gdino","yoloe"],     // model nào thấy
      "color": "white",
      "grid_cell": [3,4],
      "crop_emb_id": "crop_0091"           // ref tới Milvus instance index
    }
  ],
  "counts": {"person": 5, "car": 2},
  "count_method": {"person": "detector", "crowd": "vlm"}
}
```
- **JSONB** trong Postgres → faceted/spatial/color filter.
- **crop embeddings** → Milvus collection riêng cho instance search.

---

## 7. Validation KHÔNG có nhãn (vì solo, không annotate)

Đây là phần khó nhất khi zero-shot. 4 cách kiểm chất lượng mà không cần gán nhãn:

1. **Weak reference từ BTC:** `Objects/` (Faster-RCNN OpenImagesV4) BTC cung cấp sẵn — dùng làm reference cho ~600 lớp giao COCO. Đo agreement (IoU) giữa pipeline mới vs BTC trên các lớp chung. Lệch lớn → debug.
2. **Teacher-student spot check:** chạy **MM-GDINO-Large hoặc Qwen3-VL** (model mạnh, chậm) làm "teacher" trên **~200 keyframe random** → so với YOLOE (student) fast pass. Đo recall/precision tương đối. Không phải nhãn thật nhưng đủ để chọn threshold.
3. **Cross-detector consistency:** vùng cả YOLOE + MM-GDINO cùng vote = high-confidence (gần như chắc đúng). Vùng chỉ 1 model = cần soi. Tỉ lệ agreement là proxy cho độ tin.
4. **Eyeball qualitative (~30 phút):** render bbox lên ~100 keyframe đại diện (cháy, lũ, họp báo, giao thông…) → nhìn mắt thường. Không phải labeling — chỉ sanity check open-vocab có bắt đúng concept tin tức không, có cần chỉnh vocab §3 không.

> Mục tiêu validation: **chọn confidence threshold + chốt vocab list**, không phải đo mAP chính xác.

---

## 8. Tài nguyên & chạy (1 A100 đủ cho OD)

| Bước | Model | Throughput (A100) |
|---|---|---|
| Bulk tagging | YOLOE | ~30-50 fps |
| Concept detect | MM-GDINO | ~5-10 fps |
| Dense/đếm (on-demand) | Qwen3-VL grounding | ~0.5 fps (chỉ subset) |
| Crop re-encode | PE-Core | batch, nhanh |

~100K keyframe: YOLOE ~1h, MM-GDINO ~3-6h, crop-embed ~1h. Qwen3-VL chỉ chạy trên subset khó (~vài %). **Checkpoint mỗi 1000 keyframe.**

---

## 9. Tóm tắt — 5 quyết định cốt lõi

1. **Open-vocab thay closed-set** (YOLOE + MM-GDINO) — vì entity tin tức Việt ngoài COCO, không fine-tune được.
2. **Ensemble + WBF** — 2 detector vote, box chính xác + recall cao.
3. **Concept vocab thiết kế kỹ (§3)** — yếu tố quyết định chất lượng open-vocab, là chỗ đáng đầu tư nhất.
4. **Attributes rẻ (7×7 grid + color + count + crop-embed)** — biến detection thành feature query-able.
5. **Validation không nhãn** qua weak-reference BTC + teacher-student + cross-detector consistency + eyeball.

**Việc đầu tiên nên làm:** chạy YOLOE + MM-GDINO trên ~100 keyframe đại diện với vocab §3, render bbox, soi mắt → tinh chỉnh vocab. Đây là vòng lặp quan trọng nhất của pipeline zero-shot.
