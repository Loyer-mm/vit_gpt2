import matplotlib.pyplot as plt

# --------------------------
# 1. 准备数据
# --------------------------
epochs = [0, 1, 2, 3, 4]
val_losses = [2.9066, 2.6231, 2.4964, 2.4213, 2.3816]

# CLIPScore 数据 (Epoch 0 vs Epoch 4)
clip_epochs = ['Epoch 0', 'Epoch 4']
clip_scores = [25.7363, 26.3824]

# --------------------------
# 2. 创建画布
# --------------------------
plt.figure(figsize=(12, 5))

# --- 左图：验证集 Loss 曲线 ---
plt.subplot(1, 2, 1)
plt.plot(epochs, val_losses, marker='o', color='#2c3e50', linewidth=2, markersize=8, label='Validation Loss')
# 添加数据标注
for i, txt in enumerate(val_losses):
    plt.annotate(f'{txt:.4f}', (epochs[i], val_losses[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Validation Loss Convergence', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- 右图：CLIPScore 提升对比 ---
plt.subplot(1, 2, 2)
bars = plt.bar(clip_epochs, clip_scores, color=['#3498db', '#e74c3c'], width=0.4)
# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.title('CLIPScore Improvement', fontsize=14, fontweight='bold')
plt.ylabel('Average CLIPScore', fontsize=12)
plt.ylim(24, 28)  # 缩放 Y 轴以便观察细微差距

# --------------------------
# 3. 布局调整与保存
# --------------------------
plt.tight_layout()
plt.savefig("training_report_metrics.png", dpi=300)
print("可视化报表已保存为: training_report_metrics.png")
plt.show()