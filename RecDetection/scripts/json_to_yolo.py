import os
import json

# Ayarlar
classes = ["circle", "square", "rectangle"]  # data.yaml'daki sınıflar
datasets = {
    "train": "datasets/metals/images/train",
    "val": "datasets/metals/images/val"
}
labels_out = {
    "train": "datasets/metals/labels/train",
    "val": "datasets/metals/labels/val"
}

# Klasörleri oluştur
for key in labels_out:
    os.makedirs(labels_out[key], exist_ok=True)

# Fonksiyon: JSON → YOLO
def json_to_yolo(json_path, yolo_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in classes:
            continue
        class_id = classes.index(label)
        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_center = (x_min + x_max) / 2 / data["imageWidth"]
        y_center = (y_min + y_max) / 2 / data["imageHeight"]
        width = (x_max - x_min) / data["imageWidth"]
        height = (y_max - y_min) / data["imageHeight"]
        lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
    with open(yolo_path, "w") as f:
        f.write("\n".join(lines))

# Tüm klasörleri işle
for key in datasets:
    json_folder = datasets[key]
    label_folder = labels_out[key]
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)
            yolo_path = os.path.join(label_folder, filename.replace(".json", ".txt"))
            json_to_yolo(json_path, yolo_path)

print("JSON → YOLO dönüşümü tamamlandı!")
