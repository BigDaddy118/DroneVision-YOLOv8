import pandas as pd
from pathlib import Path

# 结果路径（根据你的实际情况调整）
result_dir = Path("runs/detect/train-6")
results_csv = result_dir / "results.csv"

# 读取并清理空列
df = pd.read_csv(results_csv)
df = df.dropna(axis=1, how='all')

# 以 mAP50-95(B) 为基准找最佳 epoch
best = df.loc[df['metrics/mAP50-95(B)'].idxmax()]

# 输出整体性能
print("=" * 50)
print("         YOLOv8n 训练最佳性能")
print("=" * 50)
print(f"最佳 Epoch       : {int(best['epoch'])}")
print(f"mAP@50           : {best['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95        : {best['metrics/mAP50-95(B)']:.4f}")
print(f"Precision        : {best['metrics/precision(B)']:.4f}")
print(f"Recall           : {best['metrics/recall(B)']:.4f}")
print(f"Val Box Loss     : {best['val/box_loss']:.4f}")
print(f"Val Cls Loss     : {best['val/cls_loss']:.4f}")
print(f"Val DFL Loss     : {best['val/dfl_loss']:.4f}")
print("=" * 50)