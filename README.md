# ViT + GPT-2 Deep Fusion Image Captioning

基于 **ViT (Vision Transformer) 编码器** 与 **GPT-2 解码器** 深度融合的图像描述（Image Captioning）模型。给定一张自然图像，模型自动生成对应的英文描述文本。

## 核心思想

传统的 Encoder-Decoder 架构仅在解码器的输入端融合一次视觉特征。本项目的"深度融合"（Deep Fusion）策略在 GPT-2 的**每一层 Transformer Block 中都插入 Cross-Attention 模块**，使文本特征在每一层都能与视觉特征进行交互，从而增强多模态信息的融合深度。

### 模型架构

```
输入图片 (3×224×224)
       │
       ▼
ViT-Base Encoder (12 层, patch_size=16)
       │
       ▼
视觉特征 (B, 197, 768)
       │
       │   输入文本 → GPT-2 Tokenizer → (B, seq_len)
       │         │
       │         ▼
       │   GPT-2 词嵌入 + 位置嵌入
       │         │
       │         ▼
       │   ┌──────────────────────────────────────┐
       │   │  DeepFusion Block × 12               │
       │   │  ┌─ Self-Attention (GPT-2 预训练)    │
       │   │  ├─ Cross-Attention (Q=文本, K/V=图像)│
       │   │  └─ MLP (GPT-2 预训练)               │
       │   └──────────────────────────────────────┘
       │         │
       │         ▼
       │   LayerNorm → LM Head → Logits
       │
       ▼
 生成描述: "a dog sitting on a couch"
```

## 项目结构

```
vit_gpt2/
├── train.py                          # 训练脚本
├── inference_fixed.py                # 推理/生成脚本
├── eval_clipscore.py                 # CLIPScore 评估
├── visualize_attention.py            # 注意力可视化
├── plot_metrics.py                   # 训练指标绘图
├── fix_image_id.py                   # 数据修复工具
├── load_dataset.py                   # 数据集下载
├── run_pipeline.sh                   # 一键训练-推理-评估流水线
├── requirements.txt                  # Python 依赖
│
├── models/
│   └── vit_gpt2.py                   # 模型定义（核心）
├── data_sets/
│   └── coco_dataset.py               # 数据集加载与预处理
│
├── data/coco2014/                    # COCO 数据（需自行下载）
│   ├── annotations/
│   │   ├── dataset_coco.json
│   │   ├── train_list.json
│   │   ├── val_list.json
│   │   └── val_list1.json
│   └── images/
│       ├── train2014/
│       └── val2014/
│
├── runs/                             # TensorBoard 日志
├── vit_gpt2_deep_fusion_epoch*.pth   # 模型权重
└── results.json                      # 推理结果
```

## 环境要求

- Python 3.8+
- CUDA (推荐，用于 GPU 训练)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本 | 用途 |
|---|---|---|
| torch | ≥1.10 | 深度学习框架 |
| torchvision | ≥0.11 | 图像预处理 |
| transformers | ≥4.20 | GPT-2 Tokenizer / 预训练权重 |
| timm | ≥0.6 | ViT 预训练模型 |
| datasets | ≥2.0 | 下载 COCO 数据集 |
| matplotlib | ≥3.5 | 可视化 |
| numpy | ≥1.21 | 数值计算 |
| Pillow | ≥9.0 | 图像处理 |
| tqdm | ≥4.64 | 进度条 |
| clip | ≥1.0 | CLIPScore 评估 |
| opencv-python | ≥4.6 | 注意力可视化 |

## 数据集

使用 **MS-COCO 2014 Karpathy Split** 数据集：

| 划分 | 图片数 | 说明 |
|---|---|---|
| Train | ~113k | train + restval |
| Val | ~5k | 验证集 |
| Test | ~5k | 测试集 |

### 数据准备

```bash
# 1. 下载数据集
python load_dataset.py

# 2. 预处理 JSON（生成 train_list.json / val_list.json）
#    在 train.py 中自动调用，或手动运行：
python -c "from data_sets.coco_dataset import preprocess_json; preprocess_json()"

# 3. 修复 image_id（如需要）
python fix_image_id.py
```

数据需放置在 `data/coco2014/` 目录下，目录结构如下：

```
data/coco2014/
├── annotations/
│   ├── dataset_coco.json          # 原始 Karpathy JSON
│   ├── train_list.json            # 预处理后的训练集
│   └── val_list.json              # 预处理后的验证集
└── images/
    ├── train2014/                 # COCO_train2014_*.jpg
    └── val2014/                   # COCO_val2014_*.jpg
```

## 使用方法

### 一键运行

```bash
bash run_pipeline.sh
```

该脚本会依次执行训练 → 推理 → CLIPScore 评估，任一失败即停止。

### 单独执行

#### 1. 训练

```bash
python train.py
```

**关键超参数**（可在 `train.py` 中修改）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size` | 16 | 批次大小 |
| `epochs` | 5 | 训练轮数 |
| `lr` | 5e-5 | AdamW 学习率 |
| `max_length` | 32 | 序列最大长度 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `num_workers` | 4 | DataLoader 工作进程数 |

训练过程使用 **Teacher Forcing** 策略，损失函数为 CrossEntropyLoss。每个 epoch 结束后保存模型权重为 `vit_gpt2_deep_fusion_epoch{epoch}.pth`。

使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir runs/
```

#### 2. 推理

```bash
python inference_fixed.py
```

加载 `vit_gpt2_deep_fusion_epoch4.pth`，对验证集随机抽取 500 张图片进行贪婪搜索生成，结果保存到 `results.json`。

生成参数：
- 策略：贪婪搜索（argmax）
- 最大长度：30 tokens
- 图像预处理：Resize(224) → ToTensor → Normalize([0.5], [0.5])

#### 3. 评估

```bash
python eval_clipscore.py
```

使用 OpenAI CLIP (ViT-B/32) 计算生成描述与原始图片的语义相似度：

- CLIP 分别编码图片和文本
- 归一化后计算余弦相似度 × 100
- 输出平均 CLIPScore

#### 4. 注意力可视化

```bash
python visualize_attention.py
```

可视化模型在预测第一个词时，Cross-Attention 对图像不同区域的关注程度，输出叠加热力图。

#### 5. 指标绘图

```bash
python plot_metrics.py
```

绘制训练过程中验证 Loss 和 CLIPScore 的变化趋势图。

## 实验结果

在 COCO 2014 验证集上训练 5 个 epoch 的结果：

| Epoch | 验证 Loss | CLIPScore |
|---|---|---|
| 0 | 2.9066 | 25.74 |
| 1 | 2.6231 | - |
| 2 | 2.4964 | - |
| 3 | 2.4213 | - |
| 4 | 2.3816 | 26.38 |

验证 Loss 持续下降，CLIPScore 有提升趋势，表明深度融合策略有效。

## 模型细节

### ViT Encoder
- 使用 `timm` 库的 `vit_base_patch16_224` 预训练权重
- 12 层 Transformer，patch size = 16×16
- 输入：224×224 RGB 图像
- 输出：197 个 tokens（1 CLS + 196 patches），每个 768 维

### GPT-2 Decoder (Deep Fusion)
- 基于 HuggingFace `GPT2LMHeadModel` 预训练权重
- 12 层自定义 `GPT2Block`，每层包含：
  - **Self-Attention**：GPT-2 原始自注意力（预训练权重初始化）
  - **Cross-Attention**：文本 Query 与图像 Key/Value 交互（随机初始化，12 heads, dim=768）
  - **MLP**：GPT-2 原始前馈网络
- 每个子层均使用 LayerNorm + 残差连接
- 词表大小：50,257

## 注意事项

1. **预处理一致性**：训练时 `coco_dataset.py` 仅做 Resize + ToTensor（无归一化），推理时额外使用了 Normalize。如需最佳效果，建议统一预处理管线。
2. **生成退化**：贪婪搜索可能导致重复短语（repetition degeneration），可尝试 beam search 或 top-k/top-p 采样改进生成质量。
3. **模型权重**：单个 checkpoint 约 910MB，5 个 epoch 共约 4.5GB。

## License

本项目仅用于学术研究目的。
