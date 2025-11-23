import os
import torch
from transformers import AutoModel, AutoTokenizer
 

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
DEVICE = torch.device("cpu")

# 🧩 Disable all CUDA + AMPP

torch.Tensor.cuda = lambda self, *a, **kw: self  # disable all .cuda() calls
torch.set_default_dtype(torch.float32)           # force float32 globally
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
 
# Patch tensor.to() to ignore unsupported dtypes
old_to = torch.Tensor.to
def safe_to(self, *args, **kwargs):
    # Remove dtype if bfloat16 or half
    if len(args) > 0 and isinstance(args[0], torch.dtype) and args[0] in [torch.bfloat16, torch.float16]:
        args = list(args)
        args[0] = torch.float32
    if 'dtype' in kwargs and kwargs['dtype'] in [torch.bfloat16, torch.float16]:
        kwargs['dtype'] = torch.float32
    return old_to(self, *args, **kwargs)
 
torch.Tensor.to = safe_to
 

# 📁 Locate Images
# -------------------------------
image_dir = "images" if os.path.isdir("images") else "."
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
 
if not images:
    print("❌ No images found in 'images' folder or current directory.")
    exit()
 
print("\n📸 Available images:")
for i, img in enumerate(images, 1):
    print(f"  {i}. {img}")
 
choice = int(input(f"\n👉 Select an image (1-{len(images)}): "))
selected_image = os.path.join(image_dir, images[choice - 1])
print(f"\n🚀 Running inference on: {selected_image} ...")
 
# -------------------------------
# 🔄 Load Model + Tokenizer
# -------------------------------
print("\n🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
 
# Force model to use float32
model = model.float()
model.eval()
 
# -------------------------------
# 🧠 Inference
# -------------------------------
prompt = "<image>\n<|grounding|>Convert this document image into readable text."
output_path = "output"
os.makedirs(output_path, exist_ok=True)
 
with torch.no_grad():
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=selected_image,
        output_path=output_path,
        base_size=512,
        image_size=512,
        crop_mode=True,
        save_results=True,
        test_compress=False,
    )
 
# 📜 Output
# -------------------------------
print("\n✅ OCR complete! Output saved in:", output_path)
print("\n🧾 Result:\n")
print(res)
