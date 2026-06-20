# datasets/coco_dataset.py
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer
import json

class CocoCaptionDataset(Dataset):
    def __init__(self, json_path, max_length=32):
        """
        json_path: str，JSON 文件路径，需要是列表形式 [{"image":..., "caption":..., "split":...}, ...]
        max_length: int，GPT2 tokenizer 最大长度
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)  # JSON 必须是列表，每个元素 {"image":..., "caption":..., "split":...}

        self.max_length = max_length

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 没有 pad_token，需要设置

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 根据文件名判断 train/val
        if item['image'].startswith("COCO_train2014"):
            img_path = f"data/coco2014/images/train2014/{item['image']}"
        else:
            img_path = f"data/coco2014/images/val2014/{item['image']}"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            item['caption'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return image, tokens.input_ids.squeeze(0)


# --------------------------
# JSON 预处理函数
# --------------------------
def preprocess_json(original_json_path, output_train_json, output_val_json):
    """
    original_json_path: 原始 Karpathy JSON
    output_train_json: 输出训练集 JSON
    output_val_json: 输出验证集 JSON
    """
    with open(original_json_path, "r") as f:
        raw = json.load(f)

    train_list = []
    val_list = []

    for img in raw["images"]:
        caption = img["sentences"][0]["raw"].strip()  # 取第一条 caption
        coco_id = img["cocoid"]  # 拿到官方 ID
        item = {
        "image": img["filename"], 
        "caption": caption, 
        "image_id": coco_id  # 存入 JSON
    }
        # 根据 split 分类
        if img.get("split") in ["train", "restval"]:
            train_list.append({"image": img["filename"], "caption": caption})
        elif img.get("split") == "val":
            val_list.append({"image": img["filename"], "caption": caption})
        # test 或其他 split 可忽略

    # 保存训练集和验证集 JSON
    with open(output_train_json, "w") as f:
        json.dump(train_list, f)
    with open(output_val_json, "w") as f:
        json.dump(val_list, f)

    print(f"训练集: {len(train_list)} 条数据，验证集: {len(val_list)} 条数据")
