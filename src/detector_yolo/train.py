from ultralytics import YOLO
from multiprocessing import freeze_support
import torch


def main():
    torch.cuda.empty_cache()

    model = YOLO("./weights/yolo11n.pt")

    results = model.train(
        data="ccpd.yaml",
        epochs=6,
        imgsz=640,
        batch=16,
        device=0,
        cache=False,
        workers=4,
        amp=True,
        plots=True,
        optimizer="AdamW",
        save_period=5,
        patience=20,
        nbs=64,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        val=True,
        single_cls=True,
    )

    metrics = model.val(split="val")

    with torch.no_grad():
        results = model("test.jpg")
        results[0].show()


if __name__ == '__main__':
    freeze_support()
    main()
