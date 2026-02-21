import os
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import csv

# ---------------- Paths ----------------
TEST_FOLDER = "crack_images/"
OUT_FOLDER = "results/"
os.makedirs(OUT_FOLDER, exist_ok=True)

CSV_FILE = os.path.join(OUT_FOLDER, "crack_measurements.csv")

# ---------------- User Settings ----------------
MODEL_PATH = "best.pt"  # Path to your YOLO model
SCALE_MM_PER_PX = 0.2   # Adjust according to your image scale (mm per pixel)
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

# ---------------- Load YOLO Model ----------------
model = YOLO(MODEL_PATH)

# ---------------- Helper Functions ----------------
def enhance_image(img):
    """Enhance image for better crack detection (CLAHE + bilateral filter)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    den = cv2.bilateralFilter(cl, d=9, sigmaColor=75, sigmaSpace=75)
    return den

def classical_crack_mask(enh):
    """Classical crack extraction using Canny + morphology."""
    v = np.median(enh)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(enh, lower, upper, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(edges, kernel, iterations=1)
    closing = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)
    bin_mask = (closing > 0).astype(np.uint8) * 255
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 50
    mask_clean = np.zeros(output.shape, dtype=np.uint8)
    if nb_components > 0:
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_clean[output == i + 1] = 255
    return mask_clean

def combine_and_refine(model_mask, classical_mask):
    """Combine model mask with classical mask and refine."""
    if model_mask is None:
        combined = classical_mask
    else:
        combined = cv2.bitwise_or(model_mask, classical_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined

def skeleton_length_and_width(mask, scale_mm_per_px):
    """Compute crack length and width in mm."""
    bw = (mask > 0).astype(np.uint8)
    skel = skeletonize(bw)
    skel_u8 = img_as_ubyte(skel)
    length_px = np.count_nonzero(skel)
    length_mm = length_px * scale_mm_per_px
    dist_transform = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    widths_px = dist_transform[skel > 0] * 2
    if len(widths_px) > 0:
        avg_width_mm = widths_px.mean() * scale_mm_per_px
        max_width_mm = widths_px.max() * scale_mm_per_px
    else:
        avg_width_mm = 0
        max_width_mm = 0
    return length_mm, avg_width_mm, max_width_mm, skel_u8

def extract_model_mask(result, orig_shape):
    """Try to extract segmentation mask from YOLO result."""
    try:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data
            if isinstance(masks, list):
                masks_arr = np.stack([m.numpy() if hasattr(m, "numpy") else np.array(m) for m in masks])
            else:
                masks_arr = masks.numpy() if hasattr(masks, "numpy") else np.array(masks)
            if masks_arr.ndim == 3:
                combined = np.any(masks_arr, axis=0).astype(np.uint8) * 255
            elif masks_arr.ndim == 2:
                combined = (masks_arr > 0).astype(np.uint8) * 255
            else:
                combined = None
            if combined is not None and combined.shape != (orig_shape[0], orig_shape[1]):
                combined = cv2.resize(combined, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            return combined
    except Exception:
        pass
    return None

# ---------------- Main Loop ----------------
with open(CSV_FILE, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Image', 'Length_mm', 'Avg_Width_mm', 'Max_Width_mm'])

    for fname in os.listdir(TEST_FOLDER):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue
        img_path = os.path.join(TEST_FOLDER, fname)
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"Could not read {img_path}")
            continue
        h, w = orig.shape[:2]

        # --- 1) Model inference ---
        results = model(img_path, conf=0.2, imgsz=1280)

        model_mask = None
        if isinstance(results, list) and len(results) > 0:
            model_mask = extract_model_mask(results[0], orig.shape)

        # --- 2) Classical mask ---
        enhanced = enhance_image(orig)
        classical_mask = classical_crack_mask(enhanced)

        # --- 3) Combine masks ---
        combined_mask = combine_and_refine(model_mask, classical_mask)

        # --- 4) Compute length and width in mm ---
        length_mm, avg_width_mm, max_width_mm, skel_img = skeleton_length_and_width(combined_mask, SCALE_MM_PER_PX)

        # --- 5) Create overlay for visualization ---
        overlay = orig.copy()
        mask_color = np.zeros_like(orig)
        mask_color[:, :, 2] = combined_mask  # red
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.5, 0)
        if skel_img is not None:
            skel_vis = np.zeros_like(orig)
            skel_vis[:, :, 1] = skel_img  # green
            overlay = cv2.addWeighted(overlay, 1.0, skel_vis, 0.9, 0)

        # --- 6) Add large, readable text with black background ---
        text_lines = [
            f"Length(mm): {length_mm:.2f}",
            f"AvgW(mm): {avg_width_mm:.2f}",
            f"MaxW(mm): {max_width_mm:.2f}"
        ]
        y0, dy = 40, 50  # start position and line spacing
        for i, line in enumerate(text_lines):
            y = y0 + i*dy
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(overlay, (5, y - text_h - 5), (5 + text_w + 5, y + 5), (0,0,0), -1)
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        # --- 7) Save results ---
        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUT_FOLDER, f"{base_name}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(OUT_FOLDER, f"{base_name}_mask.png"), combined_mask)
        if skel_img is not None:
            cv2.imwrite(os.path.join(OUT_FOLDER, f"{base_name}_skeleton.png"), skel_img)

        # --- 8) Write to CSV ---
        csvwriter.writerow([fname, f"{length_mm:.2f}", f"{avg_width_mm:.2f}", f"{max_width_mm:.2f}"])
        print(f"[{fname}] Length: {length_mm:.2f} mm, AvgW: {avg_width_mm:.2f} mm, MaxW: {max_width_mm:.2f} mm")

print("✅ All images processed. Results and CSV saved in 'results/' folder.")
