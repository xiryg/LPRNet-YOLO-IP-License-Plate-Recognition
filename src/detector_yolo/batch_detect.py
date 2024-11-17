import os

import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO("./weights/best.pt")  # 替换为你的模型路径

input_folder = "D:/PycharmProjects/LPRNet-YOLO-IP-License-Plate-Recognition/data/ccpd_data/test/"
output_folder = "D:/PycharmProjects/LPRNet-YOLO-IP-License-Plate-Recognition/data/ccpd_detect_out/"

image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
               f.lower().endswith(('.png', '.jpg'))]

for image_path in image_files:
    print(f"正在处理: {image_path}")
    results = model(image_path)  # 推理
    for i, result in enumerate(results):  # 遍历每个结果
        # 绘制图像
        annotated_frame = result.plot()
        # 保存
        save_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(save_path, annotated_frame)
        print(f"结果已保存到: {save_path}")
