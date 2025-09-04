import cv2
import os
import pandas as pd
import shutil
import io
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, session, send_file
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
# Oturum güvenliği için gizli bir anahtar 
app.secret_key = 'super-secret-key-for-session'

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
ORIGINALS_FOLDER = "static/originals"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ORIGINALS_FOLDER, exist_ok=True)

# Referans nesnenin sabit değerleri
REFERENCE_W_CM = 50.0
REFERENCE_H_CM = 70.0

# Eski koddan gelen referans piksel değerleri
REF_W_PX = 390
REF_H_PX = 537

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")
        process_type = request.form.get("process_type", "dikdortgen_olcum")

        is_new_upload = files and files[0].filename != ''

        if is_new_upload:
            # Yeni dosyalar yüklendiğinde oturumu ve klasörleri sıfırla
            session['original_images'] = []
            session['processed_images'] = []
            session.pop('combined_data', None)
            
            # Eski dosyaları temizle
            for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, ORIGINALS_FOLDER]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Hata: {file_path} silinemedi. Sebep: {e}")

            for file in files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                original_static_path = os.path.join(ORIGINALS_FOLDER, filename)
                shutil.copyfile(filepath, original_static_path)
                session['original_images'].append(filename)
            
            # İlk işlemde processed_images-orijinal görseller
            session['processed_images'] = session['original_images'][:]
        
        # SIFIRLAMA İŞLEMİ İÇİN EK KONTROL
        # Reset işlemi, bir sonraki adımda orijinal görsellerin işlenmesi için oturumu hazırlar.
        if process_type == 'reset':
            session['processed_images'] = session['original_images'][:]
            session.pop('combined_data', None)

        # İşlenecek görsel listesini belirle
        images_to_process = session.get('processed_images', [])
        combined_data = []
        new_processed_images = []

        # İşlem türü sıfırlama değilse, görselleri işle
        if process_type != 'reset':
            for current_image_filename in images_to_process:
                if current_image_filename in session['original_images']:
                    img_path = os.path.join(ORIGINALS_FOLDER, current_image_filename)
                else:
                    img_path = os.path.join(OUTPUT_FOLDER, current_image_filename)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(current_image_filename)
                outname = f"{base}_{process_type}_{timestamp}{ext}"
                outpath = os.path.join(OUTPUT_FOLDER, outname)
                
                use_min_area = request.form.get("use_min_area") == "on"
                use_morph = request.form.get("use_morph") == "on"

                processed_img = None
                
                if process_type == "gray":
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif process_type == "blur":
                    processed_img = cv2.GaussianBlur(img, (5, 5), 0)
                elif process_type == "median_blur":
                    processed_img = cv2.medianBlur(img, 5)
                elif process_type == "edges":
                    processed_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
                elif process_type == "binary":
                    _, processed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
                elif process_type == "adaptive":
                    processed_img = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                elif process_type == "otsu":
                    _, processed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif process_type == "hist":
                    plt.figure()
                    plt.title("Görsel Histogramı")
                    plt.xlabel("Piksel Değeri")
                    plt.ylabel("Frekans")
                    color = ('b', 'g', 'r')
                    for i, col in enumerate(color):
                        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                        plt.plot(hist, color=col)
                        plt.xlim([0, 256])
                    plt.savefig(outpath)
                    plt.close()
                    new_processed_images.append(outname)
                    processed_img = None
                
                # Yeni "döndürülmüş dikdörtgen ölçüm" mantığı
                elif process_type in ["dikdortgen_olcum", "boyutlandirma", "rotated_dikdortgen_olcum"]:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    if use_morph:
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    min_area_value = 15000
                    max_area_value = 300000 
                    processed_img = img.copy()
                    pixels_per_unit = None
                    
                    if process_type == "boyutlandirma":
                        if contours:
                            ref_contour = max(contours, key=cv2.contourArea)
                            _, _, ref_w_pixels, ref_h_pixels = cv2.boundingRect(ref_contour)
                            
                            if ref_w_pixels > 0 and ref_h_pixels > 0:
                                pixels_per_unit = ((ref_w_pixels / REFERENCE_W_CM) + (ref_h_pixels / REFERENCE_H_CM)) / 2.0
                    
                    # Bu değerler, referans için sabit olarak kullanılır
                    pixels_per_cm_x = REF_W_PX / REFERENCE_W_CM
                    pixels_per_cm_y = REF_H_PX / REFERENCE_H_CM

                    for i, cnt in enumerate(contours):
                        # Alan kontrolü (Tüm ölçüm tipleri için geçerli)
                        area = cv2.contourArea(cnt)
                        if use_min_area and (area < min_area_value or area > max_area_value):
                            continue
                        
                        text_to_display = ""
                        data_row = {}

                        # Metin ve dikdörtgen çizimi için her zaman bir bounding box buluyoruz
                        x, y, w, h = cv2.boundingRect(cnt)

                        if process_type == "rotated_dikdortgen_olcum":
                            # Döndürülmüş dikdörtgeni bulma
                            rect = cv2.minAreaRect(cnt)
                            (cx_f, cy_f), (w_f, h_f), angle = rect
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)

                            # Gerçek boyutları hesapla
                            w_cm = w_f / pixels_per_cm_x
                            h_cm = h_f / pixels_per_cm_y
                            area_cm2 = w_cm * h_cm

                            # Görsel üstüne çizimler
                            cv2.drawContours(processed_img, [box], 0, (0, 255, 0), 2)
                            cv2.circle(processed_img, (int(cx_f), int(cy_f)), 5, (0, 0, 255), -1)

                            # Eksenler
                            edges=[np.linalg.norm(box[1]-box[0]), np.linalg.norm(box[2]-box[1])]
                            if edges[0]>edges[1]:
                                long_dir = (box[1]-box[0])/np.linalg.norm(box[1]-box[0])
                                short_dir = (box[3]-box[0])/np.linalg.norm(box[3]-box[0])
                                half_w = edges[0] / 2
                                half_h = edges[1] / 2
                            else:
                                long_dir = (box[2]-box[1]) / np.linalg.norm(box[2]-box[1])
                                short_dir = (box[1]-box[0])/np.linalg.norm(box[1]-box[0])
                                half_w = edges[1] / 2
                                half_h = edges[0] / 2

                            # Merkez
                            cx, cy= int(cx_f), int(cy_f)

                            # Uzun eksen
                            pt1_long = (int(cx - long_dir[0] * half_w), int(cy - long_dir[1] * half_w))
                            pt2_long = (int(cx + long_dir[0] * half_w), int(cy + long_dir[1] * half_w))
                            cv2.line(processed_img, pt1_long, pt2_long, (0, 0, 255), 2)

                            # Kısa eksen
                            pt1_short = (int(cx - short_dir[0] * half_h), int(cy - short_dir[1] * half_h))
                            pt2_short = (int(cx + short_dir[0] * half_h), int(cy + short_dir[1] * half_h))
                            cv2.line(processed_img, pt1_short, pt2_short, (255, 0, 0), 2)

                            # Yazı
                            text_to_display = f"W:{w_cm:.1f} cm, H:{h_cm:.1f} cm"
                            font_scale = 0.5
                            font_thickness = 1
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (text_width, text_height) = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)[0]
                            cv2.rectangle(processed_img, (box[0][0], box[0][1] - text_height - 5),
                                        (box[0][0] + text_width, box[0][1] - 5), (0, 0, 0), -1)
                            cv2.putText(processed_img, text_to_display, (box[0][0], box[0][1] - 5),
                                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                            """ # Eksenler
                            p1 = box[0]
                            p2 = box[1]
                            p3 = box[2]
                            p4 = box[3]

                            # Eksen vektörleri
                            vector_x = p2-p1
                            vector_y = p3-p2

                            # Vektörleri normalleştirme
                            if np.linalg.norm(vector_x) > 0:
                                unit_vector_x = vector_x / np.linalg.norm(vector_x)
                            else:
                                unit_vector_x = np.array([1,0])

                            if np.linalg.norm(vector_y) > 0:
                                unit_vector_y = vector_y / np.linalg.norm(vector_y)
                            else:
                                unit_vector_y = np.array([0,1])


                            # Eksen uzunluğunu belirle (dikdörtgenin genişlik ve yüksekliği)
                            half_w = w_f / 2
                            half_h = h_f / 2

                            # X ekseni başlangıç ve bitiş noktalarını hesapla
                            start_x = (int(cx_f - unit_vector_x[0] * half_w), int(cy_f - unit_vector_x[1] * half_w))
                            end_x = (int(cx_f + unit_vector_x[0] * half_w), int(cy_f + unit_vector_x[1] * half_w))

                            # Y ekseni başlangıç ve bitiş noktalarını hesapla
                            start_y = (int(cx_f - unit_vector_y[0] * half_h), int(cy_f - unit_vector_y[1] * half_h))
                            end_y = (int(cx_f + unit_vector_y[0] * half_h), int(cy_f + unit_vector_y[1] * half_h))

                            # Eksenleri çizme (kırmızı X, mavi Y)
                            cv2.line(processed_img, start_x, end_x, (0, 0, 255), 2) # Kırmızı
                            cv2.line(processed_img, start_y, end_y, (255, 0, 0), 2) # Mavi
 """

                            """ text_to_display = f"W:{w_cm:.1f} cm, H:{h_cm:.1f} cm" """
                            data_row = {
                                "image_filename": current_image_filename,
                                "object_id": i + 1,
                                "width_px": w_f,
                                "height_px": h_f,
                                #"angle": angle,
                                "area_px": w_f * h_f,
                                "width_cm": w_cm,
                                "height_cm": h_cm,
                                "area_cm2": area_cm2
                                #"corners": box.tolist()
                            }
                        else: # Mevcut düz dikdörtgen ölçüm mantığı
                            text_to_display = f"W:{w} px, H:{h} px"
                            data_row = {"image_filename": current_image_filename, "object_id": i + 1, "x": x, "y": y, "w (pixels)": w, "h (pixels)": h}
                            
                            if process_type == "boyutlandirma" and pixels_per_unit:
                                real_w = w / pixels_per_unit
                                real_h = h / pixels_per_unit
                                text_to_display = f"W:{real_w:.2f} cm, H:{real_h:.2f} cm"
                                data_row["w (cm)"] = real_w
                                data_row["h (cm)"] = real_h

                            cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Metinleri çizme (döngünün içinde ve x, y her zaman tanımlı)
                        font_scale = 0.5
                        font_thickness = 1
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (text_width, text_height) = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)[0]
                        cv2.rectangle(processed_img, (x, y - text_height - 5), (x + text_width, y - 5), (0, 0, 0), -1)
                        cv2.putText(processed_img, text_to_display, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        combined_data.append(data_row)
                    
                if processed_img is not None:
                    cv2.imwrite(outpath, processed_img)
                    new_processed_images.append(outname)
            
            if new_processed_images:
                session['processed_images'] = new_processed_images
            
            if combined_data:
                df = pd.DataFrame(combined_data)
                session['combined_data'] = df.to_json()

    # Sayfa ilk açıldığında veya POST işlemi bittiğinde buraya gelir
    return render_template(
        "index.html",
        original_images=session.get('original_images', []),
        processed_images=session.get('processed_images', []),
        table_html=session.get('combined_data') and pd.read_json(session['combined_data']).to_html(classes="data") or None
    )

@app.route("/download_data")
def download_data():
    if 'combined_data' in session:
        df = pd.read_json(session['combined_data'])
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='tum_sonuclar.xlsx'
        )
    return "Veri bulunamadı.", 404

@app.route("/download_image/<filename>")
def download_image(filename):
    original_path = os.path.join(ORIGINALS_FOLDER, filename)
    if os.path.exists(original_path):
        return send_file(original_path, as_attachment=True)
    
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    
    return "Dosya bulunamadı.", 404

if __name__ == "__main__":
    app.run(debug=True)