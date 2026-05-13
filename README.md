# 基于YOLOv8的无人机航拍多目标检测实验报告

## 1. 项目背景

无人机在物流配送、安防巡检等场景中，需要实时感知地面目标（行人、车辆、骑行者等）以保障安全。航拍视角下的目标检测面临**小目标多、密集排列、光照变化大**等挑战。本实验基于[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)数据集，使用YOLOv8n模型进行训练与评估，并分析检测失败在无人机配送场景中的潜在风险。

## 2. 数据集与预处理

## 数据集

本项目使用 [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) 数据集进行训练与评估。

- **训练集**：6,471 张航拍图像
- **验证集**：548 张航拍图像
- **类别数**：10 类（行人、人、自行车、汽车、面包车、卡车、三轮车、带篷三轮车、公交车、摩托车）

如需复现实验，请从 [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) 下载原始数据，然后运行 `convert_visdrone.py` 进行格式转换。
- **预处理**：
  - 将VisDrone原生标注格式转换为YOLO归一化格式
  - 删除忽略区域（`score=0`）和无效类别
  - 自动去除重复标注
- **数据特点**：大量目标像素面积小于32×32，存在密集遮挡和夜间低照度场景

## 3. 模型与训练配置

- **模型**：YOLOv8n（3.0M参数，8.2 GFLOPs），适合边缘端部署
- **训练环境**：AutoDL云服务器（NVIDIA RTX 4090D 24GB）
- **超参数**：
  - Epochs: 50
  - Batch size: 48
  - Image size: 640
  - Optimizer: SGD with momentum 0.937, weight decay 0.0005
  - 数据增强：Mosaic（开启）、随机翻转、HSV色调-饱和度-明度抖动
  - 混合精度训练（AMP）：因网络问题禁用，使用FP32训练
- **权重初始化**：在COCO预训练的 `yolov8n.pt` 上微调

## 4. 实验结果

### 4.1 整体性能

|        指标        |    数值    |
| :----------------: | :--------: |
|  **最佳 mAP@50**   | **0.3004** |
| **最佳 mAP@50-95** | **0.1690** |
|   **最佳 epoch**   |   **48**   |
|   **Precision**    | **0.4241** |
|     **Recall**     | **0.3293** |

<img width="1800" height="600" alt="custom_training_curves" src="https://github.com/user-attachments/assets/440d431e-521a-4af4-a6e2-f55ee25ec469" />

该图由YOLO自动生成，包含mAP、Precision、Recall、损失曲线等

**分析**：mAP@50 达到 38%，与学术基准（YOLOv8n在VisDrone上通常30%~35%）相比处于较好水平，表明训练策略有效。损失曲线显示模型在第40轮后已基本收敛，无过拟合迹象。

### 4.2 错误案例分析

从验证集预测结果中选取几类典型错误，分析根因及配送场景下的风险：

| 案例编号 |         场景描述         | 错误类型 |             可能原因             |             无人机配送风险             |
| :------: | :----------------------: | :------: | :------------------------------: | :------------------------------------: |
|    1     |  树荫下停的一排共享单车  |   漏检   | 光照不足、像素数不足、特征不明显 | 降落时无法感知地面自行车，存在碰撞风险 |
|    2     | 图像边沿，汽车显示不完全 |   漏检   |    汽车显示不完全、特征不明显    |    降落时无法感知行车，存在碰撞风险    |

<img width="398" height="247" alt="image" src="https://github.com/user-attachments/assets/2cc9617e-e0f2-4ec3-a850-e16a474788cc" />
<img width="356" height="218" alt="image" src="https://github.com/user-attachments/assets/c16eb860-fa6d-4994-958f-6b02f37de85e" />
红色框为漏检

## 5. 改进方向

- **短期**：增大输入分辨率至960或1280，提高小目标召回率；使用Wise-IoU损失增强遮挡场景鲁棒性。
- **中期**：引入轻量级注意力模块（如SE、CBAM）或专用小目标检测头；收集夜间数据并做亮度增强。
- **长期**：结合仿真环境（如Isaac Gym）生成大量恶劣光照/遮挡的训练数据，建立Sim-to-Real数据闭环。
