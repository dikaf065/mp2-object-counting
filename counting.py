"""
Mini Project 2 — Car Counting dari Foto Aerial Parkiran
Mata Kuliah: Pengolahan Citra dan Video
Pipeline: Percentile-Adaptive HSV Segmentation + Morphological Opening + Contour Filtering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('output/steps', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ─────────────────────────────────────────
# STEP 0: Load Gambar
# ─────────────────────────────────────────
img_bgr = cv2.imread(r"C:\Users\andik\mp2-object-counting\input\parking_ori.jpg")
if img_bgr is None:
    raise FileNotFoundError("File input/parking.jpg tidak ditemukan!")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
h, w    = img_bgr.shape[:2]
img_area = h * w
print(f"[INFO] Image size: {w}x{h}, total pixels: {img_area}")

# ─────────────────────────────────────────
# STEP 1: Color Space Exploration
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].imshow(img_rgb);                     axes[0,0].set_title("Original RGB");          axes[0,0].axis("off")
axes[0,1].imshow(gray, cmap="gray");           axes[0,1].set_title("Grayscale");             axes[0,1].axis("off")
axes[0,2].imshow(img_hsv[:,:,2], cmap="gray"); axes[0,2].set_title("HSV – V (Brightness)"); axes[0,2].axis("off")
axes[1,0].imshow(img_hsv[:,:,1], cmap="hot");  axes[1,0].set_title("HSV – S (Saturation)"); axes[1,0].axis("off")
axes[1,1].imshow(img_lab[:,:,0], cmap="gray"); axes[1,1].set_title("LAB – L (Lightness)");  axes[1,1].axis("off")
axes[1,2].imshow(img_lab[:,:,1], cmap="RdBu"); axes[1,2].set_title("LAB – a (Red–Green)");  axes[1,2].axis("off")
plt.suptitle("Step 1 – Color Space Exploration", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("output/steps/step1_colorspace.png", dpi=120, bbox_inches="tight")
plt.close()
print("[STEP 1] Saved.")

# ─────────────────────────────────────────
# STEP 2: Multi-Channel HSV Segmentation
#
# Threshold dihitung dari PERCENTILE gambar (bukan nilai absolut),
# sehingga otomatis menyesuaikan resolusi & eksposur foto.
#
# Mobil putih/silver → V tinggi  (≥ percentile-85)
# Mobil hitam        → V rendah  (≤ percentile-15)
# Mobil berwarna     → S tinggi  (≥ percentile-80)
# Aspal              → V medium, S rendah → dihapus dari mask terang
# ─────────────────────────────────────────
V_ch = img_hsv[:,:,2]
S_ch = img_hsv[:,:,1]

# Hitung threshold adaptif berdasarkan distribusi pixel
v_bright = int(np.clip(np.percentile(V_ch, 85), 140, 210))
v_dark   = int(np.clip(np.percentile(V_ch, 15),  20,  80))
s_color  = int(np.clip(np.percentile(S_ch, 80),  30,  90))
print(f"[INFO] Adaptive thresholds: V_bright≥{v_bright}, V_dark≤{v_dark}, S_color≥{s_color}")

mask_bright_raw = cv2.inRange(V_ch, v_bright, 255)
mask_dark       = cv2.inRange(V_ch, 0, v_dark)
mask_color      = cv2.inRange(S_ch, s_color, 255)

# Hapus aspal (V medium, S rendah) dari mask terang
asphalt_mask    = cv2.inRange(V_ch, v_dark + 10, v_bright - 10)
asphalt_mask    = cv2.bitwise_and(asphalt_mask, cv2.bitwise_not(mask_color))
mask_bright     = cv2.bitwise_and(mask_bright_raw, cv2.bitwise_not(asphalt_mask))

mask_combined   = cv2.bitwise_or(mask_bright, mask_dark)
mask_combined   = cv2.bitwise_or(mask_combined, mask_color)

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
axes[0].imshow(mask_bright,   cmap="gray"); axes[0].set_title(f"Mask Terang\nV ≥ {v_bright}");   axes[0].axis("off")
axes[1].imshow(mask_dark,     cmap="gray"); axes[1].set_title(f"Mask Gelap\nV ≤ {v_dark}");      axes[1].axis("off")
axes[2].imshow(mask_color,    cmap="gray"); axes[2].set_title(f"Mask Berwarna\nS ≥ {s_color}");  axes[2].axis("off")
axes[3].imshow(mask_combined, cmap="gray"); axes[3].set_title("Combined Mask\n(OR dari 3 mask)"); axes[3].axis("off")
plt.suptitle("Step 2 – Multi-Channel HSV Segmentation (Percentile-Adaptive)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/steps/step2_segmentation.png", dpi=120, bbox_inches="tight")
plt.close()
print("[STEP 2] Saved.")

# ─────────────────────────────────────────
# STEP 3: Morphological Opening
#
# Kernel kecil (3x3) — hanya buang noise titik kecil.
# TIDAK pakai Closing besar karena menyebabkan mobil menyatu.
# ─────────────────────────────────────────
k_open      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_opened = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, k_open, iterations=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(mask_combined, cmap="gray"); axes[0].set_title("Sebelum Morphology\n(Combined Mask)", fontsize=12); axes[0].axis("off")
axes[1].imshow(mask_opened,   cmap="gray"); axes[1].set_title("Setelah Opening\nkernel ellipse 3x3, iter=1", fontsize=12); axes[1].axis("off")
plt.suptitle("Step 3 – Morphological Opening (Noise Removal)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/steps/step3_morphology.png", dpi=120, bbox_inches="tight")
plt.close()
print("[STEP 3] Saved.")

# ─────────────────────────────────────────
# STEP 4: Contour Detection & Filtering
#
# Area threshold RELATIF terhadap ukuran gambar (scale-adaptive),
# bukan nilai absolut — agar bekerja di semua resolusi.
#
# min_area  = 0.55% dari total pixel gambar
# max_area  = 3.3%  dari total pixel gambar
# solidity  ≥ 0.50 (buang garis marka & noise panjang-tipis)
# ─────────────────────────────────────────
min_area     = img_area * 0.0055
max_area     = img_area * 0.033
area_per_car = img_area * 0.012    # referensi luas 1 mobil
print(f"[INFO] Area thresholds: min={min_area:.0f}, max={max_area:.0f}, per_car={area_per_car:.0f}")

contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[INFO] Total raw contours: {len(contours)}")

valid = []
for c in contours:
    area = cv2.contourArea(c)
    if area < min_area or area > max_area:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    asp = bw / bh if bh > 0 else 0
    if asp < 0.25 or asp > 3.5:
        continue
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity  = area / hull_area if hull_area > 0 else 0
    if solidity < 0.50:
        continue
    valid.append((c, area, x, y, bw, bh))

print(f"[INFO] Valid contours: {len(valid)}")

# Visualisasi contour
contour_vis = img_rgb.copy()
for c, *_ in valid:
    cv2.drawContours(contour_vis, [c], -1, (0, 255, 0), 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(mask_opened, cmap="gray")
axes[0].set_title("Mask Setelah Opening", fontsize=12)
axes[0].axis("off")
axes[1].imshow(contour_vis)
axes[1].set_title(f"Contour Valid ({len(valid)} region)\nScale-adaptive | asp 0.25–3.5 | solidity ≥ 0.50", fontsize=12)
axes[1].axis("off")
plt.suptitle("Step 4 – Contour Detection & Filtering", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/steps/step4_contour.png", dpi=120, bbox_inches="tight")
plt.close()
print("[STEP 4] Saved.")

# ─────────────────────────────────────────
# STEP 5: Counting & Bounding Box Output
# ─────────────────────────────────────────
result_img = img_rgb.copy()
car_count  = 0
ref_dim    = max(w, h) * 0.15   # lebar referensi mobil ~15% dari dimensi terbesar

for c, area, x, y, bw, bh in valid:
    # Estimasi jumlah mobil dalam blob berdasarkan area
    n = max(1, round(area / area_per_car))
    # Koreksi: jika bbox terlalu lebar/tinggi → kemungkinan 2+ mobil berdampingan
    if bw > ref_dim * 1.5: n = max(n, round(bw / ref_dim))
    if bh > ref_dim * 1.3: n = max(n, round(bh / ref_dim))
    n = min(n, 3)  # maksimum 3 mobil per blob

    car_count += n
    color = (0, 220, 50) if n == 1 else (255, 140, 0)
    cv2.rectangle(result_img, (x, y), (x + bw, y + bh), color, 3)
    cv2.putText(result_img, str(n), (x + 5, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# Label total di pojok kiri atas
cv2.rectangle(result_img, (10, 10), (350, 65), (0, 0, 0), -1)
cv2.putText(result_img, f"Jumlah Mobil: {car_count}", (18, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 80), 3)

print(f"\n✅ JUMLAH MOBIL TERDETEKSI: {car_count}")

# ─────────────────────────────────────────
# STEP 6: Simpan Output Final
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 11))
axes[0].imshow(img_rgb);    axes[0].set_title("Citra Input (Original)", fontsize=14); axes[0].axis("off")
axes[1].imshow(result_img); axes[1].set_title(f"Hasil Deteksi — {car_count} Mobil", fontsize=14); axes[1].axis("off")

p1 = mpatches.Patch(color="#00dc32", label="1 mobil (individual)")
p2 = mpatches.Patch(color="#ff8c00", label="Estimasi 2+ mobil (blob gabungan)")
axes[1].legend(handles=[p1, p2], loc="lower right", fontsize=11, framealpha=0.9)

plt.suptitle("Mini Project 2 — Object Counting: Car Detection", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("output/result.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OUTPUT] output/result.png saved.")
