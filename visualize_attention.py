import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math
from models.vit_gpt2 import ViTGPT2
from data_sets.coco_dataset import CocoCaptionDataset
from transformers import GPT2Tokenizer
from PIL import Image

# 1. 配置與模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTGPT2().to(device)

# 載入你訓練好的權重 (請確保路徑正確)
checkpoint_path = "vit_gpt2_deep_fusion_epoch4.pth" 
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# --------------------------
# 2. 定位特定圖片
# --------------------------
TARGET_IMAGE_NAME = "COCO_val2014_000000120356.jpg"
# 這是你 COCO val2014 圖片存放的實際路徑
IMAGE_DIR = "data/coco2014/images/val2014/" 
image_path = os.path.join(IMAGE_DIR, TARGET_IMAGE_NAME)

if not os.path.exists(image_path):
    raise FileNotFoundError(f"找不到图片: {image_path}，请检查路径。")

# 使用與訓練一致的預處理
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

raw_image = Image.open(image_path).convert('RGB')
img_tensor = transform(raw_image).unsqueeze(0).to(device)

# --------------------------
# 3. 推理並獲取注意力
# --------------------------
# 這裡我們模擬生成第一個詞的過程
input_ids = torch.tensor([[50256]], device=device) # <|endoftext|>

with torch.no_grad():
    # 深度融合版 forward 會返回 logits 和 all_attns (12層的列表)
    logits, all_attns = model(img_tensor, input_ids)
    
    # 同時我們看看模型實際會生成什麼詞
    next_token = logits[:, -1, :].argmax(dim=-1)
    gen_word = tokenizer.decode(next_token)
    print(f"对于这张图，模型预测的第一个词是: '{gen_word.strip()}'")

# --------------------------
# 4. 提取並處理熱力圖 (取最後一層)
# --------------------------
# last_layer_attn shape: [1, heads, seq_len, 197]
last_layer_attn = all_attns[-1]
# 平均所有 Heads
attn_map = last_layer_attn.mean(dim=1) 
# 取最後一個 token 對圖像的注意力，並剔除第一個 CLS token 權重
attn_weights = attn_map[0, -1, 1:] # [196]

# 重塑為 14x14
side = int(math.sqrt(attn_weights.size(0)))
heatmap = attn_weights.reshape(side, side).cpu().numpy()

# 縮放熱力圖至 224x224
heatmap_resized = cv2.resize(heatmap, (224, 224))
# 正規化處理以便顯示
heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

# --------------------------
# 5. 繪圖與保存
# --------------------------
plt.figure(figsize=(10, 5))

# 左圖：原圖
plt.subplot(1, 2, 1)
plt.imshow(np.array(raw_image.resize((224, 224))))
plt.title("Original Image")
plt.axis('off')

# 右圖：疊加注意力熱力圖
plt.subplot(1, 2, 2)
# 顯示原圖
plt.imshow(np.array(raw_image.resize((224, 224))))
# 疊加熱力圖 (使用 jet 顏色，alpha 控制透明度)
plt.imshow(heatmap_resized, alpha=0.5, cmap='jet')
plt.title(f"Attention on: '{gen_word.strip()}'")
plt.axis('off')

save_name = f"attention_{TARGET_IMAGE_NAME}"
plt.savefig(save_name)
print(f"COCO_val2014_000000120356.jpg注意力可视化已保存为: {save_name}")
plt.show()