import cv2
import os
import matplotlib.pyplot as plt

# Kırpılmış görüntülerin ana klasörü
cropped_root_dir = "results/cropped"

# İşlenecek her klasör
for image_folder in os.listdir(cropped_root_dir):
    folder_path = os.path.join(cropped_root_dir, image_folder)
    if not os.path.isdir(folder_path):
        continue

    # İşlem türleri ve alt klasörler
    processes = {
        "binary": lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],
        "adaptive": lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        "otsu" : lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "gray": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "blur": lambda img: cv2.GaussianBlur(img, (5,5), 0),
        "median_blur": lambda img: cv2.medianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 5),
        "edges": lambda img: cv2.Canny(img, 100, 200),
        "hist": None # Özel işlemlerle yapılacak
    }

    for process_name, func in processes.items():
        process_dir = os.path.join(folder_path, process_name)
        os.makedirs(process_dir, exist_ok=True)

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)

            # Renkli veya gri histogram oluşturma
            if process_name == "hist":
                if len(img.shape) == 2: # Gri ise
                    plt.figure()
                    plt.title("Grayscale Histogram")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
                    hist = cv2.calcHist([img], [0], None, [256], [0,256])
                    plt.plot(hist)
                    plt.xlim({0,256})
                    save_path = os.path.join(process_dir, os.path.splitext(img_file)[0]+"_hist.png")
                    plt.savefig(save_path)
                    plt.close()

                else: # Renkli ise
                    color = ('b', 'g', 'r')
                    plt.figure()
                    plt.title("Color Histogram")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
                    for i, col in enumerate(color):
                        hist = cv2.calcHist([img], [i], None, [256], [0,256])
                        plt.plot(hist, color=col)
                        plt.xlim([0,256])
                    save_path = os.path.join(process_dir, os.path.splitext(img_file)[0]+"_hist.png")
                    plt.savefig(save_path)
                    plt.close()
                print(f"hist image saved: {save_path}")


            else:
                processed_img = func(img)
                save_path = os.path.join(process_dir, img_file)
                cv2.imwrite(save_path, processed_img)
                print(f"{process_name} image saved: {save_path}")

                # Eğer binary ise temizlenmiş versiyonunu da kaydet
                if process_name in ["binary", "adaptive", "otsu"]:
                    #Kernel
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                    opened = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
                    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

                    cleaned_dir = os.path.join(folder_path, f"{process_name}_cleaned")
                    os.makedirs(cleaned_dir, exist_ok=True)
                    cleaned_path = os.path.join(cleaned_dir, img_file) 
                    cv2.imwrite(cleaned_path, cleaned)
                    print(f"cleaned {process_name} image saved: {cleaned_path}")
