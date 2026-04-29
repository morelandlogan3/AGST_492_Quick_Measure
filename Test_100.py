# ============================================================
# HAY STRAW LENGTH MEASUREMENT TOOL — COMPLETE + FIXED
# ============================================================

import cv2
import numpy as np
import logging
from skimage.morphology import skeletonize
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

YOLO_MODEL_PATH = "best_K_Fold.pt"
YOLO_CONF       = 0.3

MIN_PATH_LENGTH_PX = 10
MIN_MASK_AREA_PX   = 200
MIN_SIZE_COMPONENT = 200
IOU_THRESH         = 0.7

MIN_LENGTH_MM = 10
MAX_LENGTH_MM = 1500   # raised — straw 2 was 448mm, give headroom

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================
# YOLO
# ============================================================

def load_yolo_model(path):
    from ultralytics import YOLO
    return YOLO(path)

# ============================================================
# FILE INPUT
# ============================================================

def select_files():
    return [
        r"C:\Users\Logan\Documents\VS_Code\AGST 492 project\Straw_Images\Dropbox\IMG_2733.jpg"
    ]

# ============================================================
# OVERLAP FILTER
# ============================================================

def filter_overlapping_masks(masks, iou_thresh=IOU_THRESH):
    kept = []
    for m in sorted(masks, key=lambda x: -x["confidence"]):
        keep = True
        m_bin = m["mask"] > 0
        for k in kept:
            k_bin = k["mask"] > 0
            inter = np.logical_and(m_bin, k_bin).sum()
            union = np.logical_or(m_bin, k_bin).sum()
            if union == 0:
                continue
            if inter / union > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(m)
    return kept

# ============================================================
# YOLO MASK EXTRACTION
# ============================================================

def get_masks_yolo(image, model):
    results = model(image, conf=YOLO_CONF)[0]
    masks_raw = []

    if results.masks is None:
        print("DEBUG: YOLO returned no masks at all")
        return masks_raw, results

    print(f"DEBUG: raw YOLO detections = {len(results.boxes)}")

    for i, box in enumerate(results.boxes):
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0:  # quarter — skip, only used for scaling
            continue

        mask = results.masks.data[i].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        area = np.sum(mask > 0)
        if area < MIN_MASK_AREA_PX:
            print(f"  skip mask {i}: area {area}px too small")
            continue

        masks_raw.append({
            "mask":       mask,
            "confidence": conf,
            "source":     "yolo"
        })

    print(f"DEBUG: masks before overlap filter = {len(masks_raw)}")
    masks_filtered = filter_overlapping_masks(masks_raw)
    print(f"DEBUG: masks after overlap filter  = {len(masks_filtered)}")

    return masks_filtered, results

# ============================================================
# MASK CLEANUP
# ============================================================

def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

# ============================================================
# REMOVE TINY DISCONNECTED BLOBS
# ============================================================

def merge_small_components(mask, min_size=MIN_SIZE_COMPONENT):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 255
    return cleaned

# ============================================================
# BFS — LONGEST SKELETON PATH
# ============================================================

def bfs_farthest(point_set, start):
    visited  = {start}
    queue    = deque([(start, 0.0)])
    farthest = (start, 0.0)

    while queue:
        cur, dist = queue.popleft()
        if dist > farthest[1]:
            farthest = (cur, dist)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in point_set and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + np.hypot(dx, dy)))

    return farthest

def longest_path_fast(points):
    point_set = set(map(tuple, points))
    start = next(iter(point_set))
    far1  = bfs_farthest(point_set, start)
    far2  = bfs_farthest(point_set, far1[0])
    return far2[1]

# ============================================================
# SCALE — AUTO-DETECT QUARTER COIN
# ============================================================

def get_pixel_to_mm_from_quarter(results):
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) != 0:
            continue

        if results.masks is not None:
            mask = results.masks.data[i].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            area = np.sum(mask)
            if area == 0:
                continue
            diameter_px = np.sqrt(4 * area / np.pi)
        else:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            diameter_px = max(x2 - x1, y2 - y1)

        QUARTER_DIAMETER_MM = 24.26
        scale = QUARTER_DIAMETER_MM / diameter_px
        print(f"DEBUG: quarter diameter = {diameter_px:.1f}px  →  scale = {scale:.4f} mm/px")
        return scale

    return None

# ============================================================
# MEASURE ONE MASK  (with detailed rejection reasons)
# ============================================================

def measure_mask(mask_entry, pixel_to_mm, idx=None):
    label = f"straw {idx}" if idx is not None else "straw"
    m = mask_entry["mask"]

    raw_area = np.sum(m > 0)

    # 1. clean → remove small blobs
    m = clean_mask(m)
    m = merge_small_components(m)

    post_area = np.sum(m > 0)
    if post_area == 0:
        print(f"  [{label}] REJECTED — mask empty after cleaning (raw area={raw_area}px)")
        return None

    # 2. slight dilation to bridge skeleton gaps
    kernel = np.ones((2, 2), np.uint8)
    m = cv2.dilate(m, kernel, iterations=1)

    # 3. skeletonize
    skeleton = skeletonize(m // 255)
    points   = np.column_stack(np.where(skeleton > 0))

    if len(points) < 3:
        print(f"  [{label}] REJECTED — skeleton too sparse ({len(points)} pts, post-clean area={post_area}px)")
        return None

    # 4. longest path
    length_px = longest_path_fast(points)
    length_mm = length_px * pixel_to_mm

    print(f"  [{label}] skeleton pts={len(points)}, path={length_px:.1f}px → {length_mm:.1f}mm")

    if length_px < MIN_PATH_LENGTH_PX:
        print(f"  [{label}] REJECTED — path too short in px ({length_px:.1f} < {MIN_PATH_LENGTH_PX})")
        return None

    if length_mm < MIN_LENGTH_MM:
        print(f"  [{label}] REJECTED — too short ({length_mm:.1f}mm < {MIN_LENGTH_MM}mm)")
        return None

    if length_mm > MAX_LENGTH_MM:
        print(f"  [{label}] REJECTED — too long ({length_mm:.1f}mm > {MAX_LENGTH_MM}mm)")
        return None

    return length_mm

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = load_yolo_model(YOLO_MODEL_PATH)
    files = select_files()

    results_all = []

    for f in files:
        print(f"\n{'='*50}")
        print(f"Processing: {f}")
        print('='*50)

        img = cv2.imread(f)
        if img is None:
            print(f"ERROR: could not read image — check path")
            continue

        masks, yolo_results = get_masks_yolo(img, model)

        pixel_to_mm = get_pixel_to_mm_from_quarter(yolo_results)
        if pixel_to_mm is None:
            print("WARNING: no quarter detected — using default scale (0.55 mm/px)")
            pixel_to_mm = 0.55
        else:
            print(f"Scale: {pixel_to_mm:.4f} mm/px")

        file_lengths = []
        for i, m in enumerate(masks):
            length = measure_mask(m, pixel_to_mm, idx=i+1)
            if length is not None:
                print(f"  → ACCEPTED: {length:.1f} mm  (conf={m['confidence']:.2f})")
                file_lengths.append(length)
            else:
                print(f"  → conf={m['confidence']:.2f}")

        results_all.extend(file_lengths)
        print(f"\nDetected {len(file_lengths)} / {len(masks)} straws in this image")

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print('='*50)
    for i, r in enumerate(results_all, 1):
        print(f"  {i}: {r:.1f} mm")
    print(f"\nTotal straws measured: {len(results_all)}")
    if results_all:
        print(f"Mean:    {np.mean(results_all):.1f} mm")
        print(f"Min/Max: {np.min(results_all):.1f} / {np.max(results_all):.1f} mm")