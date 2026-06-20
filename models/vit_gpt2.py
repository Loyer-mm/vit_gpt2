import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel#预训练GPT-2 LMHeadmodel自带“语言模型输出头”
from timm import create_model #timm库做视觉模型的库,加载ViT
import math

#跨模态注意力
class GPT2CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12):#(hidden size)dim=768,num_heads=12
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads#每个头部维度64维
        self.scale = self.head_dim ** -0.5 #scale是缩放因子，防止点积过大、softmax爆炸
        

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        
        
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        b, t, c = q.shape
        q = self.q(q).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(k).view(b, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(v).view(b, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
       
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        weights = self.attn_dropout(attn)
        
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(out)), attn

# 解码器Block
class GPT2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        # 这里使用 GPT2 原生的 Self-Attention 以保留预训练的语言能力
        self.attn = None 
        self.ln_2 = nn.LayerNorm(dim)
        self.cross_attn = GPT2CrossAttention(dim) # 注入你的 Cross-Attention
        self.ln_3 = nn.LayerNorm(dim)
        self.mlp = None

    def forward(self, x, enc_out):
        # self.attn(self.ln_1(x)) 返回的是 (output, past_key_values, ...)
        #GPT-2自注意力
        attn_output = self.attn(self.ln_1(x))[0] 
        x = x + attn_output
        
        # Cross-Attention
        ca_out, attn_map = self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + ca_out
        
     
        mlp_out = self.mlp(self.ln_3(x))
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        x = x + mlp_out
        return x, attn_map

# -完整的模型
class ViTGPT2(nn.Module):
    def __init__(self, depth=12):
        super().__init__()
        # 1. Encoder: ViT
        vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(depth)])
        
        # 2. Decoder: GPT2 结构
        gpt2_base = GPT2LMHeadModel.from_pretrained('gpt2')
        self.wte = gpt2_base.transformer.wte
        self.wpe = gpt2_base.transformer.wpe
        self.drop = nn.Dropout(0.1)
        self.ln_f = gpt2_base.transformer.ln_f
        self.lm_head = gpt2_base.lm_head
        
        # 3. 深度融合层
        self.h = nn.ModuleList([GPT2Block(768) for _ in range(depth)])
        
        # 4. 手动拷贝预训练权重
        for i in range(depth):
            self.h[i].attn = gpt2_base.transformer.h[i].attn
            self.h[i].mlp = gpt2_base.transformer.h[i].mlp

    def forward(self, images, input_ids):
        # Encoder 前向
        x_img = self.patch_embed(images)
        x_img = torch.cat((self.cls_token.expand(x_img.shape[0], -1, -1), x_img), dim=1)
        x_img = x_img + self.pos_embed
        for blk in self.blocks:
            x_img = blk(x_img)
            
        # Decoder 前向
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        x_txt = self.wte(input_ids) + self.wpe(pos)
        x_txt = self.drop(x_txt)
        
        all_attns = []
        for blk in self.h:
            x_txt, attn = blk(x_txt, x_img)
            all_attns.append(attn)
            
        x_txt = self.ln_f(x_txt)
        logits = self.lm_head(x_txt)
        return logits, all_attns