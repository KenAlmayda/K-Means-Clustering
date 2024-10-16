import streamlit as st
import cv2
import numpy as np
import requests
import matplotlib
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Atur Matplotlib untuk menggunakan backend non-interaktif
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses citra dari URL
@st.cache_data
def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Gagal memuat citra dari URL.")
        return image
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat citra dari URL: {e}")
        return None

# Fungsi untuk memuat dan memproses citra yang diunggah oleh pengguna
def load_and_preprocess_image(image_file):
    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        if img is None:
            raise ValueError("Gagal memuat citra yang diunggah.")
        img_resized = cv2.resize(img, (256, 256))  # Mengubah ukuran citra agar lebih efisien
        img_normalized = img_resized / 255.0  # Normalisasi intensitas piksel
        return img_resized, img_normalized
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses citra yang diunggah: {e}")
        return None, None

# Fungsi untuk ekstraksi fitur (warna dan tekstur)
def extract_features(image, use_lbp=False):
    # Pastikan citra berada dalam format uint8 untuk operasi yang tepat
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Fitur warna (RGB)
    features = image_uint8.reshape(-1, 3)
    
    # Ekstraksi fitur tekstur menggunakan Local Binary Pattern (LBP)
    if use_lbp:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_features = lbp.reshape(-1, 1)
        features = np.hstack((features, lbp_features))
    
    # Normalisasi fitur untuk clustering
    return StandardScaler().fit_transform(features)

# Fungsi untuk melakukan clustering menggunakan K-Means
def perform_clustering(features, n_clusters, max_iter, init_method):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init_method)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

# Fungsi untuk menghitung skor silhouette
def calculate_silhouette_score(features, labels):
    return silhouette_score(features, labels)

# Fungsi untuk visualisasi hasil clustering
def visualize_clustering(image, labels, n_clusters):
    clustered_image = labels.reshape(image.shape[:2])  # Bentuk ulang label menjadi ukuran citra
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Citra Asli")
    ax1.axis('off')
    ax2.imshow(clustered_image, cmap='viridis')
    ax2.set_title(f"Citra Hasil Clustering (K={n_clusters})")
    ax2.axis('off')
    return fig

# Atur judul aplikasi pada Streamlit
st.markdown("<h1 style='text-align: center; color: black;'>Aplikasi Clusterisasi Gambar Citra Udara dengan K-Means</h1>", unsafe_allow_html=True)

# Menampilkan informasi identitas kelompok dengan gaya HTML yang lebih modern
html_code = """
<div style="background-color:black;padding:10px;border-radius:10px;">
    <h3 style="color:white;text-align:center;">UTS Data Mining</h3>
    <p style="color:white;text-align:center;font-size:16px;">
        <strong>140810210028:</strong> Ken Almayda Fathurrahman<br>
        <strong>140810217001:</strong> Andrew Orisar Boekorsyom<br>
        <strong>140810220082:</strong> Jaya Goval Unedo Hutasoit
    </p>
</div>
"""
st.components.v1.html(html_code, height=150)

# Membuat sidebar untuk pemilihan sumber gambar dan pengaturan clustering
st.sidebar.markdown("<h2 style='text-align: center; color: #4CAF50;'>Pengaturan</h2>", unsafe_allow_html=True)

# Pilihan sumber citra: citra default atau unggah citra sendiri
image_source = st.sidebar.radio("Pilih Sumber Citra", ["Citra Default", "Unggah Citra Sendiri"])

# Kumpulan URL citra default
default_images_urls = {
    'Pantai': 'https://live.staticflickr.com/5643/30168153163_d3d09ab582_b.jpg',
    'Kebun Sawit': 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjiAab_iWcek216Ncmav0wu7gpevKMXIWodHEzDjUNgIsRw-FGKFsbzcX6oxOgQqj5CmCPL35AkxFRJgZWBclabELUBIlW4IcCIb7Zi4y1FtsTOvbHoXTlNE9awiabeBwD_3q9djKYPoNE/w1200-h630-p-k-no-nu/1.png',
    'Kota': 'https://asset.kompas.com/data/photo/2014/07/21/1932244Perbatasan-tegal-Brebes780x390.JPG',
    'Bandara': 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiLnSAr8Kk6zQEXVMJetvndiKbbzAmqJRjdZigBb50cATqA0W-Ar7mW176F1NugvtUZgel9zhzT9_2TyOLcr3s5upSHwgSiSxrPkDXT11caolgKgUHpHXSLJ6C6LPIfDXItsadFSfs5gJ8a/s1600/Hartsfield-Jackson-Atlanta_airport.jpg',
    'Gurun': 'https://rm.id/images/berita/med/citra-satelit-ungkap-pangkalan-rudal-nuklir-israel-dekat-yerusalem_66414.jpg'
}

image = None  # Variabel inisialisasi citra
normalized_image = None

# Jika pengguna memilih citra default
if image_source == "Citra Default":
    selected_image_name = st.sidebar.selectbox("Pilih citra default", list(default_images_urls.keys()))
    selected_image_url = default_images_urls[selected_image_name]
    image = load_image_from_url(selected_image_url)
    if image is not None:
        st.image(image, channels="BGR", caption=f"Citra Dipilih: {selected_image_name}", use_column_width=True)
        normalized_image = image / 255.0  # Normalisasi citra default

# Jika pengguna memilih untuk mengunggah citra sendiri
elif image_source == "Unggah Citra Sendiri":
    uploaded_image = st.sidebar.file_uploader("Unggah citra udara", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image, normalized_image = load_and_preprocess_image(uploaded_image)
        if image is not None:
            st.image(image, channels="BGR", caption="Citra Udara yang Diunggah", use_column_width=True)

# Parameter untuk clustering
n_clusters = st.sidebar.slider("Jumlah Klaster yang Diinginkan", 2, 10, 3)
max_iter = st.sidebar.slider("Maksimal Iterasi K-Means", 100, 1000, 300)
init_method = st.sidebar.selectbox("Metode Inisialisasi K-Means", ['k-means++'])
use_lbp = st.sidebar.checkbox("Gunakan fitur tekstur")

# Melakukan clustering jika citra telah dipilih atau diunggah
if normalized_image is not None:
    # Ekstraksi fitur dari citra
    features = extract_features(normalized_image, use_lbp)

    # Lakukan clustering
    labels, kmeans = perform_clustering(features, n_clusters, max_iter, init_method)

    # Visualisasikan hasil clustering
    fig = visualize_clustering(image, labels, n_clusters)
    st.pyplot(fig)

    # Hitung dan tampilkan silhouette score
    score = calculate_silhouette_score(features, labels)
    st.write(f"Skor Silhouette: {score:.2f}")
    st.write("Penjelasan Skor Silhouette: Skor mendekati 1 menunjukkan klaster yang lebih baik terdefinisi. "
             "Skor di atas 0.5 umumnya menunjukkan struktur klaster yang kuat.")

    # Jika pengguna ingin membandingkan dengan ground truth
    ground_truth = st.file_uploader("Unggah data ground truth (opsional)", type=["txt", "csv"])
    if ground_truth is not None:
        try:
            gt_labels = np.loadtxt(ground_truth, delimiter=',')
            ari_score = adjusted_rand_score(gt_labels, labels)
            st.write(f"Adjusted Rand Index (perbandingan dengan ground truth): {ari_score:.2f}")
            st.write("Penjelasan ARI Score: Skor berkisar dari -1 hingga 1, di mana 1 menunjukkan kesesuaian sempurna "
                     "antara clustering dan ground truth.")
        except Exception as e:
            st.error(f"Kesalahan saat memuat file ground truth: {e}")
else:
    st.warning("Silakan pilih atau unggah citra sebelum melanjutkan.")
