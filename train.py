import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.vit_gpt2 import ViTGPT2  
from data_sets.coco_dataset import CocoCaptionDataset, preprocess_json
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os


# TensorBoard
writer = SummaryWriter("runs/vit_gpt2_deep_fusion")


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意：这里确保你的 ViTGPT2 类已经按照我之前给你的“深度融合版”修改过了
model = ViTGPT2().to(device)


# 预处理 JSON 
if not os.path.exists("data/coco2014/annotations/train_list.json"):
    print("正在预处理 JSON...")
    preprocess_json(
        original_json_path="data/coco2014/annotations/dataset_coco.json",
        output_train_json="data/coco2014/annotations/train_list.json",
        output_val_json="data/coco2014/annotations/val_list.json"
    )


# 加载训练集和验证集
# 确保 CocoCaptionDataset 返回的是 (image, input_ids)
train_set = CocoCaptionDataset(json_path="data/coco2014/annotations/train_list.json")
val_set = CocoCaptionDataset(json_path="data/coco2014/annotations/val_list.json")

# 获取 tokenizer 方便后续处理 padding
tokenizer = train_set.tokenizer 

# 使用 collate_fn 来处理 batch 内部的 padding 填充
# 如果你之前的 dataset.py 里有这个函数，请确保它被正确调用
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

print(f"训练集大小: {len(train_set)}, batch 数: {len(train_loader)}")
print(f"验证集大小: {len(val_set)}, batch 数: {len(val_loader)}")


# 优化器和损失函数
# 使用稍小的学习率来微调预训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 【关键修改】ignore_index=-100 
# 这会让损失函数忽略掉我们设置的特殊位置，防止模型只学会“预测停止”
loss_fn = CrossEntropyLoss(ignore_index=-100)


# 训练循环
num_epochs = 5
step = 0



for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, input_ids in pbar:
        images = images.to(device)
        input_ids = input_ids.to(device)

        labels = input_ids.clone()
        # 将所有的 PAD token 替换为 -100，Loss 会自动忽略它们
        labels[labels == tokenizer.pad_token_id] = -100
        
        # 模型前向传播
        # 输入去掉最后一个 token
        logits, _ = model(images, input_ids[:, :-1])

        # 损失函数计算：labels 要去掉第一个 token (因为 logits 预测的是第2个开始的词)
        # 形状对齐：logits -> [Batch*Seq, Vocab], labels -> [Batch*Seq]
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)), 
            labels[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止训练炸掉
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 更新进度条
        pbar.set_postfix(loss=loss.item())
        writer.add_scalar("train/loss", loss.item(), step)
        step += 1


    # 验证集评价
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for images, input_ids in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            logits, _ = model(images, input_ids[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            val_loss_total += loss.item()
            
    avg_val_loss = val_loss_total / len(val_loader)
    print(f"Epoch {epoch} - 平均验证损失: {avg_val_loss:.4f}")
    writer.add_scalar("val/loss", avg_val_loss, epoch)


    # 保存模型
    torch.save(model.state_dict(), f"vit_gpt2_deep_fusion_epoch{epoch}.pth")

writer.close()
print("训练完成！建议现在运行 inference 看看效果。")