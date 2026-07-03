import os
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# Konfigurasi Halaman 
st.set_page_config(page_title="Sobat Sampah", layout="wide", initial_sidebar_state="collapsed")


# Load Model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Pastikan file 'best.pt' berada di direktori yang sama dengan script ini
    model_path = os.path.join(base_dir, 'best.pt') 
    return YOLO(model_path)

model = load_model()

# Basis Data Deskripsi Sampah Sesuai Kelas Model
waste_info = {
    "Botol Teh pucuk": {"type": "ANORGANIK - PLASTIK", "desc": "Botol plastik PET. Kosongkan, remukkan, dan buang ke tempat sampah daur ulang."},
    "Bungkus Paket": {"type": "ANORGANIK - PLASTIK", "desc": "Plastik pembungkus paket. Buang ke tempat sampah anorganik."},
    "Bungkus Tisu": {"type": "ANORGANIK - PLASTIK", "desc": "Plastik kemasan tisu. Buang ke tempat sampah anorganik."},
    "Eskrim": {"type": "ANORGANIK - PLASTIK/KARTON", "desc": "Kemasan es krim. Bersihkan dari sisa makanan lalu buang ke tempat sampah anorganik."},
    "Hello Panda Merah": {"type": "ANORGANIK - KERTAS", "desc": "Kemasan karton makanan ringan. Pisahkan dari plastik di dalamnya, buang ke tempat sampah kertas."},
    "Hello Panda Pink": {"type": "ANORGANIK - KERTAS", "desc": "Kemasan karton makanan ringan. Pisahkan dari plastik di dalamnya, buang ke tempat sampah kertas."},
    "Kertas": {"type": "KERTAS", "desc": "Pastikan kertas tidak basah/berminyak sebelum didaur ulang."},
    "Oreo": {"type": "ANORGANIK - PLASTIK", "desc": "Bungkus makanan ringan (multilayer). Buang ke tempat sampah anorganik."},
    "Plastik": {"type": "ANORGANIK - PLASTIK", "desc": "Sampah plastik umum. Buang ke tempat sampah anorganik."},
    "Plastik bening": {"type": "ANORGANIK - PLASTIK", "desc": "Plastik transparan. Bisa didaur ulang jika bersih."},
    "Stela": {"type": "ANORGANIK - PLASTIK", "desc": "Kemasan pengharum ruangan. Buang ke tempat sampah anorganik."},
    "Tisu": {"type": "RESIDU / ORGANIK", "desc": "Tisu kotor/berminyak masuk ke residu. Tisu basah air biasa dapat dikomposkan."},
    "default": {"type": "TIDAK DIKENALI", "desc": "Kategori tidak spesifik. Buang sesuai jenis material dominannya."}
}

# Header UI
st.title("Sobat Sampah")
st.markdown("---")

# Layout 2 Kolom Kiri-Kanan (Otomatis menyesuaikan layar di mobile)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input Kamera")
    # PERBAIKAN: Menambahkan lebar target untuk widget kamera
    img_buffer = st.camera_input("Pindai Sampah Anda", width=640)


with col2:
    st.markdown("### Hasil Deteksi")
    st.markdown("Output")
    if img_buffer is not None:

        # Konversi
        image = Image.open(img_buffer).convert("RGB")
        # image = image.resize((640, 640))
        width, height = image.size
        
        # PERBAIKAN: Menggunakan Center Crop langsung ke 640x640 (Bukan di-resize)
        # Menghitung koordinat potong bagian tengah gambar
        left = (width - 640) / 2
        top = (height - 640) / 2
        right = (width + 640) / 2
        bottom = (height + 640) / 2
        image = image.crop((left, top, right, bottom))
        
        # Inferensi Model YOLO
        results = model.predict(image, imgsz=640)
        result = results[0]
        
        if len(result.boxes) > 0:
            # Ambil deteksi dengan confidence tertinggi untuk detail utama
            best_box = result.boxes[0]
            best_class_id = int(best_box.cls[0].item())
            best_class_name = model.names[best_class_id]
            info = waste_info.get(best_class_name, waste_info["default"])
            
            res_plotted_bgr = result.plot(line_width=2, font_size=12)
            
            # PERBAIKAN: Konversi warna dari BGR ke RGB agar warna kembali normal/benar
            res_plotted_rgb = res_plotted_bgr[:, :, ::-1]
            # Tampilkan Gambar dengan warna yang benar
            st.image(res_plotted_rgb, caption="Visualisasi Deteksi (640x640)", width=400)
            
            # Detail Utama
            st.success(f"Kategori Dominan: **{info['type']}**")
            st.write(f"**Objek:** {best_class_name}")
            st.write(f"**Tindakan:** {info['desc']}")
            
            # Tabel Confidence Seluruh Objek Terdeteksi
            detected_data = []
            if result.boxes is not None:
                for box in result.boxes:
                    # Mengambil nilai scalar langsung menggunakan .item() setelah konversi ke CPU
                    cls_id = int(box.cls.cpu().item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf.cpu().item()) * 100
                    
                    detected_data.append({
                        "Objek": cls_name, 
                        "Keyakinan (%)": f"{conf:.2f}"
                    })
            
            # Validasi jika data terisi, baru buat dan tampilkan tabelnya
            if detected_data:
                df = pd.DataFrame(detected_data)
                st.table(df)
            else:
                st.write("Detail akurasi objek gagal diproses.")
        else:
            st.warning("Tidak ada objek sampah yang terdeteksi secara meyakinkan. Sesuaikan pencahayaan dan coba lagi.")
    else:
        st.info("Menunggu input dari kamera...")
