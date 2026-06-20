import torch
import clip
from PIL import Image
import json
import os
from tqdm import tqdm

# 1. 配置路径
RESULTS_JSON = "results.json"
# 原始 COCO 验证集图片存放目录
IMAGE_DIR = "data/coco2014/images/val2014/" 


# 2. 初始化 CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

def calculate_clip_score(image_path, text):
    """计算单张图片与对应文本的相似度"""
    if not os.path.exists(image_path):
        return None
    
    # 预处理图片和文本
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # 截断文本防止超过 CLIP 的 77 token 限制
    text_input = clip.tokenize([text[:200]], truncate=True).to(device)

    with torch.no_grad():
        # 提取特征
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_input)

        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
        # CLIPScore 通常定义为：相似度 * 100 (有些实现会加一个缩放系数)
        similarity = (image_features @ text_features.T).item()
        score = max(similarity, 0) * 100
        
    return score


# 3. 执行评估
def main():
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    scores = []
    print(f"正在计算 {len(results)} 条结果的 CLIPScore...")

    for item in tqdm(results):
        img_id = item["image_id"]
        caption = item["caption"]
        
        # 匹配 COCO 文件名格式 (COCO_val2014_000000XXXXXX.jpg)
        img_filename = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        score = calculate_clip_score(img_path, caption)
        
        if score is not None:
            scores.append(score)
        else:
            print(f"Warning: 找不到图片 {img_path}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print("\n" + "="*30)
        print(f"评估完成！")
        print(f"样本数量: {len(scores)}")
        print(f"平均 CLIPScore: {avg_score:.4f}")
        print("="*30)
    else:
        print("没有成功计算任何分数，请检查图片路径。")

if __name__ == "__main__":
    main()