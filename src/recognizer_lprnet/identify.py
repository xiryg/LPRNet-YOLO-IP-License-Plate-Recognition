import torch
import cv2
import numpy as np
from model.LPRNet import build_lprnet
from utils.locate_lp import detect_and_save_roi
from data.load_data import CHARS, transform

img_path = "./data/test.jpg"
model_path = "./weights/best.pt"
img = detect_and_save_roi(img_path, model_path)
img = cv2.resize(img, (94, 24))
img = transform(img)
img = img[np.newaxis, :]
ims = torch.Tensor(img)

lprnet = build_lprnet(lpr_max_len=8, phase=True, class_num=68, dropout_rate=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lprnet = lprnet.to(device)
lprnet.load_state_dict(torch.load("./weights/Final_LPRNet_model.pth", weights_only=True))

prebs = lprnet(ims.to(device))
prebs = prebs.cpu().detach().numpy()
prebs_labels = list()

for i in range(prebs.shape[0]):
    preb = prebs[i, :, :]
    preb_label = list()

    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))

    no_repeat_blank_label = list()
    pre_c = preb_label[0]

    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)

    # 去除重复字符和空白字符
    for c in preb_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    prebs_labels.append(no_repeat_blank_label)

for label in prebs_labels:
    lb = ""
    print(label)
    for item in label:
        lb += CHARS[item]
    print(f"Predicted license plate : {lb}")
