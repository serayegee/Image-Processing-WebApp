from ultralytics import YOLO

# Model seç (yolov8n -> en küçük ve hızlı, yolov8m/l -> daha güçlü)
model = YOLO("yolov8n.pt")  

# Eğit
model.train(
    data="data.yaml",   # dataset ayarı 
    epochs=50,          # eğitim turu
    imgsz=640,          # resim boyutu
    batch=16,           # batch size
    name="metal_shapes" # çıktılar runs/train/metal_shapes/ içine kaydedilir
)
