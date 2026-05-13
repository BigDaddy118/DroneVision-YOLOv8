from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 训练参数
    results = model.train(
        data='visdrone.yaml',
        epochs=50,
        imgsz=640,          # 可以尝试 960 或 1280，但会占更多显存
        batch=48,           # 尝试从 32 增大到 48，若显存溢出再调回 32 或 40
        device=0,
        amp=False,          # 禁用 AMP 以避免去 GitHub 下载 yolo26n.pt
        workers=8,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=False,
        close_mosaic=10,
        # 以下可根据需要开启
        # resume=True,      # 断点续训时解开
        # multi_scale=True, # 多尺度训练
    )
    
    # 导出最佳模型到 ONNX（可选）
    # model.export(format='onnx')

if __name__ == '__main__':
    main()