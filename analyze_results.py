import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil

# 结果目录（根据实际情况修改）
result_dir = Path("runs/detect/train-6")
results_csv = result_dir / "results.csv"
weights_dir = result_dir / "weights"

# 1. 读取训练指标
df = pd.read_csv(results_csv)
df = df.dropna(axis=1, how='all')  # 去除空列

# 找出最佳 mAP@50-95 的 epoch
best_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
print("=== 训练完成，最佳模型性能 ===")
print(f"最佳 Epoch: {int(best_epoch['epoch'])}")
print(f"mAP@50: {best_epoch['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")
print(f"Box Loss: {best_epoch['val/box_loss']:.4f}")
print(f"Cls Loss: {best_epoch['val/cls_loss']:.4f}")

# 2. 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.title('Validation mAP')

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.tight_layout()
plt.savefig(result_dir / 'custom_training_curves.png', dpi=150)
print("自定义训练曲线已保存到 custom_training_curves.png")

# 3. 失败案例分析（如果有保存的验证批次图片）
val_img_dir = result_dir
pred_imgs = list(val_img_dir.glob("val_batch*_pred.jpg"))
if pred_imgs:
    print(f"\n找到 {len(pred_imgs)} 张验证预测图，前3张路径：")
    for img in pred_imgs[:3]:
        print(img)
    print("你可以人工检查这些图片，找出误检/漏检场景")
else:
    print("\n未找到 val_batch*_pred.jpg，可能训练时未生成，可重新运行验证生成。")