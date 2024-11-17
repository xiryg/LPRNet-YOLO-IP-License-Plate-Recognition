from ultralytics import YOLO

model = YOLO("./weights/best.pt")

results = model("test.jpg")

result = results[0]

result.show()
