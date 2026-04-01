# ============================================================
# Hay Straw Length Measurement Workflow
# ============================================================
# Objectives Covered:
#   PRIMARY:
#     - Detect individual straw pieces from images
#     - Accurately measure straw length in millimeters
#     - Export results to Excel/CSV
#   SECONDARY:
#     - Classify straw as straight or curved
#     - Detect and flag overlapping/low-confidence measurements
#     - Batch processing of multiple images
#   METHODOLOGY:
#     - HSV color segmentation with interactive trackbars
#     - Morphological noise reduction
#     - Watershed overlap handling
#     - Skeletonization + graph representation
#     - Endpoint detection (true endpoints preferred, fallback to max distance)
#     - Dijkstra path length for curved straw
#     - Calibration factor (pixel-to-mm)
#     - Overlay visualization with endpoints, paths, overlap regions
# ============================================================

import cv2
import numpy as np
import os
import pandas as pd
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
import heapq
from tkinter import Tk, filedialog
import logging
import sys
import time

# ============================================================
# CONFIGURATION
# ============================================================

PIXEL_TO_MM = 0.1           # Default calibration factor (mm per pixel)
MIN_CONTOUR_AREA = 100      # Minimum pixel area to consider a contour
CURVE_THRESHOLD = 1.1       # Ratio of path/euclidean above which straw is "Curved"
DEBUG_MODE = True           # Set False to suppress debug image windows
LOG_FILE = "straw_debug.log"

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

def debug_step(name, image, wait=200):
    """Show an intermediate image window during processing if DEBUG_MODE is on."""
    if DEBUG_MODE:
        cv2.imshow(f"DEBUG: {name}", image)
        cv2.waitKey(wait)

# ============================================================
# FILE PICKER
# ============================================================

def select_files():
    log.info("Opening file picker...")
    root = Tk()
    root.withdraw()
    files = list(filedialog.askopenfilenames(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("JPG files", "*.jpg")]
    ))
    log.info(f"Selected {len(files)} file(s): {files}")
    return files

# ============================================================
# CALIBRATION
# ============================================================

def get_calibration():
    """
    Prompt user for pixel-to-mm calibration factor.
    Press Enter to use the default PIXEL_TO_MM constant.
    """
    print("\n--- CALIBRATION ---")
    print(f"Default pixel-to-mm ratio: {PIXEL_TO_MM}")
    user_input = input("Enter calibration (mm per pixel) or press Enter to use default: ").strip()
    if user_input:
        try:
            cal = float(user_input)
            log.info(f"User calibration set: {cal} mm/pixel")
            return cal
        except ValueError:
            log.warning("Invalid calibration input. Using default.")
    log.info(f"Using default calibration: {PIXEL_TO_MM} mm/pixel")
    return PIXEL_TO_MM

# ============================================================
# HSV PICKER
# ============================================================

def get_hsv(sample_path):
    """
    Interactive HSV tuning window with real-time callback updates.
    Every slider move instantly redraws the mask and overlay.
    Press 'q' or 's' to confirm and continue.
    """
    log.info(f"Opening HSV tuning window on: {sample_path}")

    img = cv2.imread(sample_path)
    if img is None:
        log.error(f"Could not read image: {sample_path}")
        sys.exit(1)
    img = cv2.resize(img, (800, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    WIN = "HSV Tuning  (press Q to confirm)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 700)

    # Shared state dict updated by callbacks
    state = {
        "H Min": 0,   "H Max": 179,
        "S Min": 0,   "S Max": 255,
        "V Min": 0,   "V Max": 255,
    }

    def update(_=None):
        """Called instantly on every slider change — redraws all panels."""
        lower = np.array([state["H Min"], state["S Min"], state["V Min"]])
        upper = np.array([state["H Max"], state["S Max"], state["V Max"]])

        mask = cv2.inRange(hsv, lower, upper)

        # Darkened overlay — only straw stays bright
        overlay = img.copy()
        overlay[mask == 0] = (40, 40, 40)

        # Colour the masked region green so it's easy to spot
        green_highlight = img.copy()
        green_highlight[mask > 0] = (0, 220, 80)
        blended = cv2.addWeighted(img, 0.5, green_highlight, 0.5, 0)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Top row: original | darkened overlay
        top = np.hstack([img, overlay])
        # Bottom row: green highlight | binary mask
        bottom = np.hstack([blended, mask_bgr])

        # Label each panel
        for panel_img, label, pos in [
            (top,    "Original",        (10, 25)),
            (top,    "Masked Overlay",  (810, 25)),
            (bottom, "Green Highlight", (10, 25)),
            (bottom, "Binary Mask",     (810, 25)),
        ]:
            cv2.putText(panel_img, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel_img, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        combined = np.vstack([top, bottom])
        # Scale down to fit screen (combined is 1600x1200)
        display = cv2.resize(combined, (800, 600))
        cv2.imshow(WIN, display)

        # Live readout in terminal
        pixel_count = int(np.sum(mask > 0))
        print(f"\r  HSV [{state['H Min']:3d}-{state['H Max']:3d}] "
              f"[{state['S Min']:3d}-{state['S Max']:3d}] "
              f"[{state['V Min']:3d}-{state['V Max']:3d}]  "
              f"Masked pixels: {pixel_count:6d}", end="", flush=True)

    def make_callback(key):
        """Returns a callback that writes to state[key] and redraws."""
        def cb(val):
            state[key] = val
            update()
        return cb

    # Create trackbars — each one triggers update() on change
    trackbar_defs = [
        ("H Min", 179, 0),
        ("H Max", 179, 179),
        ("S Min", 255, 0),
        ("S Max", 255, 255),
        ("V Min", 255, 0),
        ("V Max", 255, 255),
    ]
    for name, maxv, default in trackbar_defs:
        cv2.createTrackbar(name, WIN, default, maxv, make_callback(name))

    print("\n--- HSV TUNING ---")
    print("Move sliders — mask updates instantly.")
    print("Press 'q' or 's' when done.\n")

    # Draw initial state
    update()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), ord('s'), 27):  # q, s, or Escape
            break

    cv2.destroyAllWindows()
    print()  # newline after live readout

    lower = np.array([state["H Min"], state["S Min"], state["V Min"]])
    upper = np.array([state["H Max"], state["S Max"], state["V Max"]])
    log.info(f"HSV Lower: {lower}  |  HSV Upper: {upper}")
    return lower, upper

# ============================================================
# DIJKSTRA PATH LENGTH
# ============================================================

def dijkstra(graph, start, end):
    """Shortest path through skeleton graph — accurate for curved straw."""
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    while queue:
        dist, current = heapq.heappop(queue)
        if current == end:
            return dist
        for neighbor in graph[current]:
            new_dist = dist + np.linalg.norm(np.array(current) - np.array(neighbor))
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return float('inf')

# ============================================================
# WATERSHED SEGMENTATION
# ============================================================

def apply_watershed(mask):
    """Separate touching/overlapping straw using watershed."""
    log.debug("Applying watershed segmentation...")
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask, sure_fg)

    markers, num_markers = ndi.label(sure_fg)
    log.debug(f"Watershed markers found: {num_markers}")
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    separated = np.zeros_like(mask)
    separated[markers > 1] = 255

    debug_step("Watershed Result", separated)
    return separated, markers

# ============================================================
# BRANCH POINT DETECTION
# ============================================================

def remove_branch_points(points):
    """
    Identify skeleton branch points (pixels with >2 neighbors).
    Branch points indicate intersections or overlapping straw.
    Returns clean_points (no branches) and branch_points separately.
    """
    point_set = set(map(tuple, points))
    clean_points = []
    branch_points = []

    neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for p in points:
        neighbor_count = sum(
            1 for dp in neighbors_8
            if (p[0]+dp[0], p[1]+dp[1]) in point_set
        )
        if neighbor_count > 2:
            branch_points.append(tuple(p))
        else:
            clean_points.append(tuple(p))

    log.debug(f"  Branch points: {len(branch_points)}  |  Clean points: {len(clean_points)}")
    return clean_points, branch_points

# ============================================================
# TRUE ENDPOINT DETECTION
# ============================================================

def find_true_endpoints(points):
    """
    Find true skeleton endpoints: pixels with exactly 1 neighbor.
    Falls back to the max-distance pair if fewer than 2 endpoints found.
    Returns: (start, end, method_used)
    """
    point_set = set(map(tuple, points))
    neighbors_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    endpoints = []

    for p in points:
        neighbor_count = sum(
            1 for dp in neighbors_8
            if (p[0]+dp[0], p[1]+dp[1]) in point_set
        )
        if neighbor_count == 1:
            endpoints.append(p)

    if len(endpoints) >= 2:
        log.debug(f"  True endpoints found: {len(endpoints)}")
        pts = np.array(endpoints)
        dist_matrix = cdist(pts, pts)
        idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        return tuple(pts[idx[0]]), tuple(pts[idx[1]]), "true_endpoint"
    else:
        log.debug("  Fewer than 2 true endpoints — falling back to max distance pair.")
        pts = np.array(points)
        dist_matrix = cdist(pts, pts)
        idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        return tuple(pts[idx[0]]), tuple(pts[idx[1]]), "fallback_max_dist"

# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(image_path, hsv_lower, hsv_upper, pixel_to_mm):
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        log.error(f"Failed to load image: {image_path}")
        return [], None

    image = cv2.resize(image, (800, 600))
    debug_step("1 - Original Image", image)

    # --- Step 1: HSV Color Segmentation ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    log.debug(f"Step 1 - HSV mask white pixels: {np.sum(mask > 0)}")
    debug_step("2 - HSV Mask Raw", mask)

    # --- Step 2: Noise Reduction (Morphological Open + Close) ---
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    log.debug(f"Step 2 - After morphology white pixels: {np.sum(mask > 0)}")
    debug_step("3 - After Morphology", mask)

    # --- Step 3: Watershed Overlap Separation ---
    mask, markers = apply_watershed(mask)
    debug_step("4 - After Watershed", mask)

    # --- Step 4: Contour Detection ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log.info(f"Step 4 - Total contours: {len(contours)}")

    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    log.info(f"Step 4 - Valid contours (area >= {MIN_CONTOUR_AREA}): {len(valid_contours)}")

    contour_debug = image.copy()
    cv2.drawContours(contour_debug, valid_contours, -1, (0, 255, 255), 1)
    debug_step("5 - Valid Contours", contour_debug)

    overlay = image.copy()
    results = []

    for i, cnt in enumerate(valid_contours):
        log.debug(f"\n  --- Straw candidate {i} ---")
        area = cv2.contourArea(cnt)
        log.debug(f"  Contour area: {area:.1f} px²")

        # --- Skeletonize ---
        obj_mask = np.zeros_like(mask)
        cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
        skeleton = skeletonize(obj_mask // 255)
        skeleton = (skeleton * 255).astype(np.uint8)

        skeleton_pixels = int(np.sum(skeleton > 0))
        log.debug(f"  Skeleton pixels: {skeleton_pixels}")

        if skeleton_pixels < 2:
            log.warning(f"  Straw {i}: Skeleton too short (<2 px), skipping.")
            continue

        points = np.column_stack(np.where(skeleton > 0))

        # --- Branch Point Removal (overlap detection) ---
        clean_points, branch_points = remove_branch_points(points)
        overlap_flag = len(branch_points) > 0

        if len(clean_points) < 2:
            log.warning(f"  Straw {i}: Too few clean points after branch removal, skipping.")
            continue

        # --- Endpoint Detection ---
        start, end, endpoint_method = find_true_endpoints(clean_points)
        log.debug(f"  Endpoint method: {endpoint_method} | Start: {start} | End: {end}")

        # --- Build Skeleton Graph ---
        pts = np.array(clean_points)
        graph = {}
        for p in pts:
            tp = tuple(p)
            graph[tp] = [
                tuple(q) for q in pts
                if 0 < np.linalg.norm(p - q) <= np.sqrt(2)
            ]

        # --- Length Calculation ---
        euclid = np.linalg.norm(np.array(start) - np.array(end))
        if euclid == 0:
            log.warning(f"  Straw {i}: Zero euclidean distance, skipping.")
            continue

        path_length = dijkstra(graph, start, end)
        if path_length == float('inf'):
            log.warning(f"  Straw {i}: Dijkstra returned inf (disconnected skeleton). Using euclidean fallback.")
            path_length = euclid

        ratio = path_length / euclid
        shape = "Curved" if ratio > CURVE_THRESHOLD else "Straight"
        log.debug(f"  Euclidean: {euclid:.1f}px | Path: {path_length:.1f}px | Ratio: {ratio:.3f} | Shape: {shape}")

        # --- Confidence Scoring ---
        if overlap_flag and endpoint_method == "fallback_max_dist":
            confidence = "LOW"
        elif overlap_flag or endpoint_method == "fallback_max_dist":
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

        length_mm = path_length * pixel_to_mm
        log.info(f"  Straw {i}: {length_mm:.2f} mm | {shape} | Overlap: {overlap_flag} | Confidence: {confidence}")

        # --- Draw Overlay ---
        draw_color = (0, 0, 255) if overlap_flag else (0, 255, 0)

        # Draw skeleton pixels
        for p in clean_points:
            overlay[p[0], p[1]] = (0, 200, 255)

        # Draw endpoints
        cv2.circle(overlay, (start[1], start[0]), 5, draw_color, -1)
        cv2.circle(overlay, (end[1], end[0]), 5, draw_color, -1)

        # Draw straight-line reference between endpoints
        cv2.line(overlay, (start[1], start[0]), (end[1], end[0]), (200, 0, 200), 1)

        # Mark branch/overlap points in cyan
        for bp in branch_points:
            cv2.circle(overlay, (bp[1], bp[0]), 3, (0, 255, 255), -1)

        # Label length on image
        cx = int((start[1] + end[1]) / 2)
        cy = int((start[0] + end[0]) / 2) - 8
        cv2.putText(overlay, f"#{i} {length_mm:.1f}mm [{confidence}]",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)

        results.append({
            "Image":                  os.path.basename(image_path),
            "Straw_ID":               i,
            "Length_mm":              round(length_mm, 3),
            "Shape":                  shape,
            "Overlap_Detected":       overlap_flag,
            "Confidence":             confidence,
            "Endpoint_Method":        endpoint_method,
            "Skeleton_Pixels":        skeleton_pixels,
            "Branch_Points":          len(branch_points),
            "Path_Euclidean_Ratio":   round(ratio, 4),
            "Pixel_to_mm_Used":       pixel_to_mm,
        })

    log.info(f"Image complete: {len(results)} straw(s) measured.")
    debug_step("6 - Final Overlay", overlay, wait=800)
    return results, overlay

# ============================================================
# SUMMARY STATS
# ============================================================

def print_summary(df):
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    print(f"Total straw measured :  {len(df)}")
    if len(df) > 0:
        print(f"Average length (mm)  :  {df['Length_mm'].mean():.2f}")
        print(f"Min length (mm)      :  {df['Length_mm'].min():.2f}")
        print(f"Max length (mm)      :  {df['Length_mm'].max():.2f}")
        print(f"Straight             :  {(df['Shape'] == 'Straight').sum()}")
        print(f"Curved               :  {(df['Shape'] == 'Curved').sum()}")
        print(f"Overlap detected     :  {df['Overlap_Detected'].sum()}")
        print(f"HIGH confidence      :  {(df['Confidence'] == 'HIGH').sum()}")
        print(f"MEDIUM confidence    :  {(df['Confidence'] == 'MEDIUM').sum()}")
        print(f"LOW confidence       :  {(df['Confidence'] == 'LOW').sum()}")
    print("="*60 + "\n")

# ============================================================
# SAFE EXCEL SAVE (handles file-locked-by-Excel error)
# ============================================================

def safe_save_excel(df, filename):
    """
    Try to save the Excel file. If it is locked (open in Excel),
    automatically save to a timestamped filename instead.
    Returns the path that was actually used.
    """
    base_path = os.path.abspath(filename)
    try:
        df.to_excel(base_path, index=False)
        return base_path
    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(base_path)
        new_path = f"{name}_{timestamp}{ext}"
        log.warning(f"Permission denied on '{base_path}' — file may be open in Excel.")
        log.warning(f"Saving to '{new_path}' instead.")
        print(f"\n  WARNING: '{os.path.basename(base_path)}' is open in Excel.")
        print(f"  Saving to '{os.path.basename(new_path)}' instead.\n")
        df.to_excel(new_path, index=False)
        return new_path

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()
    log.info("=== Straw Measurement Tool Started ===")

    # 1. Select image files
    files = select_files()
    if not files:
        log.error("No files selected. Exiting.")
        sys.exit(0)

    # 2. Set calibration factor
    pixel_to_mm = get_calibration()

    # 3. Tune HSV color range on first image
    hsv_lower, hsv_upper = get_hsv(files[0])

    all_results = []

    # 4. Batch process all selected images
    for idx, f in enumerate(files):
        log.info(f"\nImage {idx+1}/{len(files)}: {f}")
        res, overlay = process_image(f, hsv_lower, hsv_upper, pixel_to_mm)
        all_results.extend(res)

        if overlay is not None:
            # Save overlay next to original image
            base = os.path.splitext(os.path.basename(f))[0]
            out_dir = os.path.dirname(os.path.abspath(f))
            out_img_path = os.path.join(out_dir, f"overlay_{base}.jpg")
            cv2.imwrite(out_img_path, overlay)
            log.info(f"Overlay image saved to: {out_img_path}")

            cv2.imshow(f"Result [{idx+1}/{len(files)}]: {os.path.basename(f)}", overlay)
            cv2.waitKey(800)

    cv2.destroyAllWindows()

    # 5. Export results
    df = pd.DataFrame(all_results)

    if len(df) > 0:
        output_xlsx = safe_save_excel(df, "straw_results.xlsx")
        output_csv  = os.path.abspath("straw_results.csv")
        df.to_csv(output_csv, index=False)
        print_summary(df)
        log.info(f"Excel saved : {output_xlsx}")
        log.info(f"CSV saved   : {output_csv}")
        print(f"Results saved to:\n  Excel : {output_xlsx}\n  CSV   : {output_csv}")
    else:
        log.warning("No straw detected in any image. Check HSV settings or MIN_CONTOUR_AREA.")
        print("\nNo straw detected. Try adjusting HSV values or lowering MIN_CONTOUR_AREA.")

    elapsed = time.time() - start_time
    log.info(f"=== Total processing time: {elapsed:.2f}s ===")
    print(f"\nDebug log saved to: {os.path.abspath(LOG_FILE)}")
