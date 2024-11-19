import torch
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from src.recognizer_lprnet.model.LPRNet import build_lprnet
from src.recognizer_lprnet.data.load_data import CHARS, transform

model_path = "../weights/best.pt"

lprnet = build_lprnet(lpr_max_len=8, phase=True, class_num=68, dropout_rate=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lprnet = lprnet.to(device)
lprnet.load_state_dict(torch.load("../weights/Final_LPRNet_model.pth", weights_only=True))


def recognize_license_plate(image):
    if image is None:
        return "请上传图片"

    def detect_and_save_roi(model_path, image):
        model = YOLO(model_path)

        # 模型推理
        results = model(image)
        result = results[0]  # 获取第一个结果对象

        # 遍历检测框
        for i, box in enumerate(result.boxes.xyxy):  # xyxy格式的检测框
            x1, y1, x2, y2 = map(int, box.tolist())  # 提取左上角和右下角坐标
            print(f"检测框 {i + 1}: 左上角({x1}, {y1}), 右下角({x2}, {y2})")

            # 提取检测框区域
            roi = image[y1:y2, x1:x2]

            return roi

    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img = detect_and_save_roi(model_path, img)

    # 预处理图片
    img = cv2.resize(img, (94, 24))
    img = transform(img)
    img = img[np.newaxis, :]
    ims = torch.Tensor(img)

    # 模型推理
    with torch.no_grad():
        prebs = lprnet(ims.to(device))
        prebs = prebs.cpu().detach().numpy()

    preb = prebs[0, :, :]
    preb_label = [np.argmax(preb[:, j], axis=0) for j in range(preb.shape[1])]

    no_repeat_blank_label = []
    pre_c = preb_label[0]

    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)

    for c in preb_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c

    license_plate = "".join(CHARS[item] for item in no_repeat_blank_label)
    return f"车牌号为： {license_plate}"


img_examples = ["https://img1.tucang.cc/api/image/show/f01aeb08e27576fc839d92bd04f7192c",
                "https://img1.tucang.cc/api/image/show/a798e119dc2f4d0d29716a24cfa8727e",
                "https://img1.tucang.cc/api/image/show/bb85289cf1eda0ee890ff82328add699"]

iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="上传车牌图片", height=300, width=700),
    outputs=gr.Textbox(label="识别结果"),
    title="CCPD-LPR",
    description="上传一张图片，包含车牌，如示例······",
    examples=img_examples
)

if __name__ == "__main__":
    iface.launch()
