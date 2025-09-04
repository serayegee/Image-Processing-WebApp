import os

# Label klasörünüzü buraya yazın
label_folder = r"C:\Users\PC\OneDrive\Masaüstü\detection\datasets\metals\labels"

# Alt klasörler: train ve val
subfolders = ["train", "val"]

for sub in subfolders:
    folder_path = os.path.join(label_folder, sub)
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Dosyayı oku
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            # Her satırın başındaki class id'yi 0 yap
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    parts[0] = "0"  # class id'yi 0 yap
                    new_lines.append(" ".join(parts) + "\n")
            
            # Dosyayı tekrar yaz
            with open(file_path, "w") as f:
                f.writelines(new_lines)

print("Tüm label dosyaları güncellendi. Class id = 0 olarak ayarlandı.")
