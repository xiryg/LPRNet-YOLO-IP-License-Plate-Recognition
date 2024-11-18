import cv2
from ultralytics import YOLO


def detect_and_save_roi(image_path, model_path):
    model = YOLO(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        return

    results = model(image_path)
    result = results[0]  # 获取第一个结果对象
    result.show()
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        print(f"检测框 : 左上角({x1}, {y1}), 右下角({x2}, {y2})")

        roi = image[y1:y2, x1:x2]

        return roi
