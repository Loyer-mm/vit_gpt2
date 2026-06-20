import json
import random
import torch
import os
from models.vit_gpt2 import ViTGPT2
from data_sets.coco_dataset import CocoCaptionDataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 统一使用 50256 作为起始和填充
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token

# 2. 加载深度融合模型
# 深度融合版 ViTGPT2 内部已经集成了 Encoder 和手写的 GPT2Block 
model = ViTGPT2().to(device)

# 深度融合权重文件
checkpoint_path = "vit_gpt2_deep_fusion_epoch4.pth" 
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"成功加载权重: {checkpoint_path}")
else:
    print(f"警告：未找到权重文件 {checkpoint_path}，请检查路径。")

model.eval()

# 3. 图像预处理 (必须与项目文件一致)
# 确保 Normalize 参数与 train.py/dataset.py 完全一致
inference_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 4. 加载数据集
dataset = CocoCaptionDataset(
    json_path="data/coco2014/annotations/val_list1.json"
)

# 随机抽取 500 个样本
indices = random.sample(range(len(dataset)), 500)

# 5. 推理生成逻辑 
@torch.no_grad()
def generate_caption(model, image_tensor, max_len=30):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    # 初始输入为 BOS (50256)
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    generated_tokens = []

    for i in range(max_len):
        # 深度融合版 model 内部会自动处理 ViT 提取特征并与 input_ids 进行 Cross-Attention
        # logits 形状为 (1, seq_len, vocab_size)
        logits, _ = model(image_tensor, input_ids)
        
        # 取最后一个词的预测
        next_token_logits = logits[:, -1, :]
        
        # 贪婪搜索：取概率最大的词 ID
        next_token_id = next_token_logits.argmax(-1).unsqueeze(0) 
        
        token_val = next_token_id.item()
        
        # 遇到结束符 EOS 则停止
        if token_val == tokenizer.eos_token_id:
            break
            
        generated_tokens.append(token_val)
        
        # 将新词拼接到 input_ids，作为下一步的输入
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # 解码
    caption = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return caption

# 6. 执行推理并保存
results = []
os.makedirs("vis_results", exist_ok=True)

print(f"开始生成预测结果...")

for idx in tqdm(indices):
    image, _ = dataset[idx] 
    real_id = dataset.data[idx]['image_id']

    # 生成描述
    caption = generate_caption(model, image)

    # 可视化前 5 张
    if len(results) < 5:
        plt.figure(figsize=(8, 6))
        # 逆归一化显示图片
        img_show = image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        plt.imshow(img_show)
        plt.title(f"Image ID: {real_id}\nCaption: {caption}")
        plt.axis("off")
        plt.savefig(f"vis_results/res_{real_id}.png")
        plt.close()

    results.append({
        "image_id": int(real_id),
        "caption": caption
    })

# 7. 保存 JSON
output_file = "results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"完成！生成 500 条结果，已保存至 {output_file}。")