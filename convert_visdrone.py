import os
from pathlib import Path

# VisDrone 类别映射 (忽略背景类0)
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}

def convert_annotation(ann_path, img_w, img_h, out_path):
    """将单个 VisDrone 标注文件转为 YOLO 格式并保存"""
    with open(ann_path, 'r') as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue

        bbox_left = float(parts[0])
        bbox_top = float(parts[1])
        bbox_w = float(parts[2])
        bbox_h = float(parts[3])
        score = int(parts[4])
        category = int(parts[5])

        # 跳过忽略区域 (score=0) 和无效类别
        if score == 0 or category not in CATEGORY_MAP:
            continue

         # 转换为 YOLO 归一化格式
        x_center = (bbox_left + bbox_w / 2) / img_w
        y_center = (bbox_top + bbox_h / 2) / img_h
        norm_w = bbox_w / img_w
        norm_h = bbox_h / img_h

        class_id = CATEGORY_MAP[category]
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def main():
    base = Path("data/VisDrone")
    
    for split in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
        ann_dir = base / split / "annotations"
        img_dir = base / split / "images"
        out_dir = base / split / "labels"   # YOLO 要求标签放在 labels 文件夹
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 假设图片是 .jpg 格式，根据实际情况可能为 .png
        img_exts = ['.jpg', '.png']
        for img_file in img_dir.iterdir():
            if img_file.suffix not in img_exts:
                continue
            
            # 匹配同名的 .txt 标注文件
            ann_file = ann_dir / (img_file.stem + ".txt")
            if not ann_file.exists():
                print(f"警告: 找不到标注 {ann_file}")
                continue
            
            # 为了获取图片宽高，我们用 OpenCV 读一下（需安装 opencv-python）
            try:
                import cv2
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"无法读取图片 {img_file}")
                    continue
                h, w = img.shape[:2]
            except ImportError:
                # 如果没装 opencv，先做个预转换（图片尺寸可不用，但转换脚本可以后续完善）
                print("OpenCV 未安装，无法获取图片尺寸。请先 pip install opencv-python")
                return
            
            out_file = out_dir / (img_file.stem + ".txt")
            convert_annotation(ann_file, w, h, out_file)
        
        print(f"✅ {split} 转换完成，标签已保存到 {out_dir}")

if __name__ == "__main__":
    main()