import json
NB = "/home/bao/Projects/OCR/hunyuanocr_colab_a100.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
def split(s): return s.splitlines(keepends=True)

old = 'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU?!")\n'
add = (old +
'\n'
'# --- Fix 404 "additional_chat_templates": tải hết model về cache rồi load OFFLINE ---\n'
'import os, huggingface_hub\n'
'from huggingface_hub import snapshot_download\n'
'print("Tải model về cache (1 lần)...")\n'
'snapshot_download(MODEL_ID)                      # public, không cần token\n'
'huggingface_hub.constants.HF_HUB_OFFLINE = True\n'
'os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
'print("Đã cache xong -> chuyển OFFLINE (tránh gọi additional_chat_templates).")\n')

patched = False
for c in nb["cells"]:
    if c["cell_type"] == "code" and any(old == s for s in c["source"]):
        j = "".join(c["source"]).replace(old, add, 1)
        c["source"] = split(j)
        patched = True
        break
assert patched, "GPU print line not found"

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
nb2 = json.load(open(NB, encoding="utf-8"))
for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"<cell {i}>", "exec")
print("OK: offline-cache fix added to Load-model cell; all cells compile")
