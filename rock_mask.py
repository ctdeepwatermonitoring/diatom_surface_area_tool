import os
import re
import glob
import math
import cv2
import numpy as np
import pandas as pd

# Cutting board is 12x18 inches -> area in square cm
length_in = 18
width_in = 12
length_cm = length_in * 2.54
width_cm = width_in * 2.54
board_area_square_cm = length_cm * width_cm


def clean_mask_by_component(mask, min_area):
    num_labels_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_inv = cv2.bitwise_not(mask)
    num_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)

    components = []
    for i in range(1, num_labels_w):
        components.append({
            'label': i,
            'area': int(stats_w[i, cv2.CC_STAT_AREA]),
            'color': 255,
            'labels': labels_w,
        })
    for i in range(1, num_labels_b):
        components.append({
            'label': i,
            'area': int(stats_b[i, cv2.CC_STAT_AREA]),
            'color': 0,
            'labels': labels_b,
        })

    components.sort(key=lambda x: x['area'])

    cleaned = mask.copy()

    for comp in components:
        if comp['area'] < min_area:
            new_value = 0 if comp['color'] == 255 else 255
            cleaned[comp['labels'] == comp['label']] = new_value

    return cleaned


def clean_mask_until_converged(mask, min_area):
    while True:
        cleaned = clean_mask_by_component(mask, min_area)
        num_labels_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        mask_inv = cv2.bitwise_not(cleaned)
        num_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)

        small_found = False
        for i in range(1, num_labels_w):
            if stats_w[i, cv2.CC_STAT_AREA] < min_area:
                small_found = True
                break
        for i in range(1, num_labels_b):
            if stats_b[i, cv2.CC_STAT_AREA] < min_area:
                small_found = True
                break

        if not small_found:
            return cleaned
        mask = cleaned


def choose_mask_by_color(hsv):
    """Return mask and label choosing between green and orange thresholds based on pixel counts."""
    lower_green = np.array([30, 80, 130])
    upper_green = np.array([90, 255, 255])
    lower_orange = np.array([0, 100, 85])
    upper_orange = np.array([20, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    if np.count_nonzero(green_mask) > np.count_nonzero(orange_mask):
        return green_mask, 'green'
    return orange_mask, 'orange'


def process_folder(image_dir, ground_truth_path, out_csv, display=False):
    # Read ground truth Excel
    gt_df = pd.read_excel(ground_truth_path)

    # Collect image paths
    patterns = [os.path.join(image_dir, '*.' + ext) for ext in ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(p))
    image_paths = sorted(image_paths)

    results = []
    scale = 0.2
    min_area_ratio = 0.001

    for img_path in image_paths:
        print('\nProcessing', img_path)
        img = cv2.imread(img_path)
        if img is None:
            print('  Could not read image, skipping')
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask, chosen = choose_mask_by_color(hsv)
        print(f'  Chosen color: {chosen}')

        total_pixels = mask.size
        min_area = int(total_pixels * min_area_ratio)
        converged_cleaned = clean_mask_until_converged(mask, min_area)

        cnts = cv2.findContours(converged_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(contours) == 0:
            print('  No contours found, skipping')
            continue

        board_mask = np.zeros_like(converged_cleaned)
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(board_mask, [main_contour], -1, 255, thickness=cv2.FILLED)

        board_pixels = int(np.count_nonzero(board_mask))
        rock_pixels = int(np.count_nonzero((board_mask == 255) & (converged_cleaned == 0)))

        rock_area_square_cm = (rock_pixels / board_pixels) * board_area_square_cm if board_pixels > 0 else float('nan')

        # Extract SID (5 digits at start) and dup flag if '-D' present
        fname = os.path.basename(img_path)
        m = re.match(r'^(\d{5})', fname)
        sid = m.group(1) if m else None
        dup_flag = 1 if ('-D' in fname or '-d' in fname) else 0

        # Match ground truth row
        gt_val = float('nan')
        if sid is not None and 'SID' in gt_df.columns:
            candidates = gt_df[gt_df['SID'].astype(str).str.strip().str.replace(r'\.0+$','', regex=True).str.extract(r'(\d+)', expand=False) == sid]
            if not candidates.empty:
                if dup_flag == 1 and 'Dup' in candidates.columns:
                    cand = candidates[candidates['Dup'] == 1]
                    if cand.empty:
                        cand = candidates
                else:
                    if 'Dup' in candidates.columns:
                        cand = candidates[(candidates['Dup'] != 1) | (candidates['Dup'].isna())]
                        if cand.empty:
                            cand = candidates
                    else:
                        cand = candidates
                if 'Average Foil Size (cm2)' in cand.columns:
                    try:
                        gt_val = float(cand.iloc[0]['Average Foil Size (cm2)'])
                    except Exception:
                        gt_val = float('nan')

        abs_err = abs(rock_area_square_cm - gt_val) if (not math.isnan(rock_area_square_cm) and not math.isnan(gt_val)) else float('nan')
        pct_err = (abs_err / gt_val * 100) if (not math.isnan(abs_err) and not math.isnan(gt_val) and gt_val != 0) else float('nan')

        results.append({
            'image_path': img_path,
            'sid': sid,
            'dup_flag': dup_flag,
            'board_pixels': board_pixels,
            'rock_pixels': rock_pixels,
            'rock_area_cm2': rock_area_square_cm,
            'ground_truth_cm2': gt_val,
            'abs_error': abs_err,
            'pct_error': pct_err,
            'chosen_color': chosen,
        })

        if display:
            resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            resized_converged = cv2.resize(converged_cleaned, (0, 0), fx=scale, fy=scale)
            cv2.imshow('Original', resized_img)
            cv2.imshow('Converged Cleaned Mask', resized_converged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f'Wrote results to {out_csv}')
        valid = df[~df['abs_error'].isna()]
        if not valid.empty:
            print(f'Processed {len(df)} images. Matched to ground truth: {len(valid)}')
            print(f"Mean absolute error (cm2): {valid['abs_error'].mean():.3f}")
            print(f"Mean percent error: {valid['pct_error'].mean():.2f}%")
    else:
        print('No results produced.')


if __name__ == '__main__':
    IMAGE_DIR = r'/home/deepuser/Documents/OneDrive_1_9-25-2025/Surface Area Photos'
    GROUND_TRUTH = r'/home/deepuser/Documents/OneDrive_1_9-25-2025/FoilWeights_2025.xlsx'
    OUT_CSV = r'/home/deepuser/Documents/OneDrive_1_9-25-2025/rock_area_results.csv'
    DISPLAY = True

    process_folder(IMAGE_DIR, GROUND_TRUTH, OUT_CSV, display=DISPLAY)