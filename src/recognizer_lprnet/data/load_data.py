from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


def transform(img):
    # img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, label_file, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.imgSize = imgSize
        self.lpr_max_len = lpr_max_len

        # 读取标注文件
        self.img_labels = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_name, label = line.split('\t')
                    self.img_labels[img_name] = label

        # 获取图片路径
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # 只保留有标注的图片
        self.img_paths = [p for p in self.img_paths if os.path.basename(p) in self.img_labels]
        random.shuffle(self.img_paths)

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        basename = os.path.basename(filename)
        # 读取图片
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        height, width, _ = Image.shape
        if height != self.imgSize[1] or width != self.imgSize[0]:
            Image = cv2.resize(Image, self.imgSize)
        Image = self.PreprocFun(Image)

        # 获取标签
        label_str = self.img_labels[basename]
        label = []
        for c in label_str:
            label.append(CHARS_DICT[c])

        return Image, label, len(label)
