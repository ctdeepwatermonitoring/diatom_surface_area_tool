import cv2
import numpy as np
import pandas as pd
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster import AgglomerativeClustering
import os
import re
import math

# Cutting board is 12x18 inches -> area in square cm
length_in = 18
width_in = 12
length_cm = length_in * 2.54
width_cm = width_in * 2.54
board_area_square_cm = length_cm * width_cm

#Function to fond the nearest pair of contours and the two closest points between them
def nearest_pair(contours):
    best = None
    best_dist = float('inf')

    for i in range(len(contours)):
        c1 = contours[i].reshape(-1, 2)
        for j in range(i + 1, len(contours)):
            c2 = contours[j].reshape(-1, 2)

            min_dist = float('inf')
            best_p1 = None
            best_p2 = None

            for p1 in c1:
                for p2 in c2:
                    d = np.linalg.norm(p1 - p2)
                    if d < min_dist:
                        min_dist = d
                        best_p1 = p1
                        best_p2 = p2

            if min_dist < best_dist:
                best_dist = min_dist
                best = (best_p1, best_p2)

    return best


#Function to close the mask using alpha shapes for each contour separately
def close_mask_per_contour(mask, alpha=0.02):
    result = np.zeros_like(mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        # Build a mini mask for just this contour, filled
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Get the actual foreground pixels belonging to this contour's region
        pts = np.column_stack(np.where(contour_mask > 0))
        pts_xy = pts[:, ::-1]  # to (x, y)
        
        if len(pts_xy) < 4:
            continue

        if len(pts_xy) > 500000:
            pts_xy = pts_xy[::100]
        
        shape = alphashape.alphashape(pts_xy, alpha)
        
        if shape is None or shape.is_empty:
            continue
        
        # Handle both Polygon and MultiPolygon results
        polys = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
        
        for poly in polys:
            if isinstance(poly, Polygon) and not poly.is_empty:
                exterior = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(result, [exterior], 255)
    
    return result

# Return mask and label choosing between blue and orange thresholds based on pixel counts.
def choose_board_mask_by_color(hsv):
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([112, 255, 255])
    lower_orange = np.array([0, 100, 85])
    upper_orange = np.array([20, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    if np.count_nonzero(blue_mask) > np.count_nonzero(orange_mask):
        return blue_mask, 'blue'
    return orange_mask, 'orange'

def choose_rock_mask_by_color(hsv, reference_mask=None, chosen_board_color=None, color_hint=None):
    """Choose rock mask. If `color_hint` is provided (e.g. 'yellow', 'red', 'purple' or folder name like 'yellow_crayon'),
    prefer that mask over automatic selection.
    """
    lower_purple = np.array([110, 0, 20])
    upper_purple = np.array([160, 100, 150])
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 40, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 10, 5])
    upper_yellow = np.array([40, 255, 255])

    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    if chosen_board_color == 'orange':
        masks = [purple_mask, yellow_mask]
        names = ['purple', 'yellow']
    else:
        masks = [purple_mask, red_mask, yellow_mask]
        names = ['purple', 'red', 'yellow']

    # If a reference mask is provided, apply it to all candidate masks
    if reference_mask is not None:
        masks = [cv2.bitwise_and(mask, reference_mask) for mask in masks]

    # If a folder-based color hint is given, normalize and prefer that mask
    if color_hint:
        hint = color_hint.lower()
        hint = hint.replace('-','_')
        hint = hint.replace('_crayon','').replace('crayon','')
        hint = hint.split('_')[0]
        for idx, name in enumerate(names):
            if name == hint:
                return masks[idx], names[idx]

    # Fallback: choose by pixel counts
    counts = np.array([np.count_nonzero(mask) for mask in masks])
    best_idx = int(np.argmax(counts))

    return masks[best_idx], names[best_idx]

# Function to remove small components based on area, followed by opening and closing the image
def initial_mask_cleaning(mask, min_area):
    #Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Bit flip white parts of the mask that are below min_contour_area
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)

    #Opening the image to remove noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    #Closing the image to connect the mask and fill in holes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41,41))
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)

    return closed_mask

def process_folder(image_dir, ground_truth_path, out_csv, init_board_mask_dir, init_rock_mask_dir, out_board_mask_dir, out_rock_mask_dir, display=False):
    # Read ground truth Excel
    gt_df = pd.read_excel(ground_truth_path)

    # Ensure mask directories exist
    os.makedirs(init_board_mask_dir, exist_ok=True)
    os.makedirs(init_rock_mask_dir, exist_ok=True)
    os.makedirs(out_board_mask_dir, exist_ok=True)
    os.makedirs(out_rock_mask_dir, exist_ok=True)

    results = []
    scale = 0.2
    min_area_ratio = 0.0002

    # Walk through image_dir and its subfolders (subfolders are crayon color folders)
    for root, dirs, files in os.walk(image_dir):
        rel_dir = os.path.relpath(root, image_dir)
        for img_name in files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                continue
            img_path = os.path.join(root, img_name)
            print('\nProcessing', img_path)
            img = cv2.imread(img_path)
            if img is None:
                print('Could not read image, skipping')
                continue

            # Choose board mask color
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            board_mask, board_chosen = choose_board_mask_by_color(hsv)
            print(f'Chosen color: {board_chosen}')

            # Prepare output subdirectory to mirror input structure
            subfolder = rel_dir if rel_dir != '.' else ''
            init_board_subdir = os.path.join(init_board_mask_dir, subfolder)
            init_rock_subdir = os.path.join(init_rock_mask_dir, subfolder)
            out_board_subdir = os.path.join(out_board_mask_dir, subfolder)
            out_rock_subdir = os.path.join(out_rock_mask_dir, subfolder)
            os.makedirs(init_board_subdir, exist_ok=True)
            os.makedirs(init_rock_subdir, exist_ok=True)
            os.makedirs(out_board_subdir, exist_ok=True)
            os.makedirs(out_rock_subdir, exist_ok=True)

            # Save initial board mask
            init_board_mask_path = os.path.join(init_board_subdir, img_name)
            cv2.imwrite(init_board_mask_path, board_mask)

            # Defining minimum contour area to filer out small contours
            total_pixels = board_mask.size
            min_contour_area = min_area_ratio * total_pixels

            # Clean initial board mask
            cleaned_board_mask = initial_mask_cleaning(board_mask, min_contour_area)

            # Use close_mask_per_contour method to close the mask
            cleaned_board_mask = close_mask_per_contour(cleaned_board_mask, alpha=0.005)

            # Save the cleaned board mask
            out_board_mask_path = os.path.join(out_board_mask_dir, img_name)
            cv2.imwrite(out_board_mask_path, cleaned_board_mask)

            # Determine folder-based color hint and skip ambiguous combined folders
            folder_name = os.path.basename(root)
            if 'purple_yellow' in folder_name.lower():
                print('Skipping ambiguous folder', folder_name)
                continue

            # Choose rock mask color (prefer folder hint)
            rock_mask, rock_chosen = choose_rock_mask_by_color(hsv, cleaned_board_mask, board_chosen, color_hint=folder_name)
            print(f'Chosen color: {rock_chosen}')

            # Save initial rock mask
            init_rock_mask_path = os.path.join(init_rock_mask_dir, img_name)
            cv2.imwrite(init_rock_mask_path, rock_mask)

            # Clean initial rock mask
            cleaned_rock_mask = initial_mask_cleaning(rock_mask, min_contour_area)

            # Use agglomerative clustering to connect the crayon parts
            #Getting closed contours
            closed_contours, closed_hierarchy = cv2.findContours(cleaned_rock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Getting the centroids of the closed contours
            valid_contours = []
            contour_centroids = []
            for contour in closed_contours:
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                contour_centroids.append((cx, cy))
                valid_contours.append(contour)

            #Clustering centroids into 5 groups
            centroids_array = np.array(contour_centroids)
            model = AgglomerativeClustering(
                n_clusters=5,
                metric="euclidean",
                linkage="ward"
            )
            labels = model.fit_predict(centroids_array)

            #Connect the contours in the same cluster by drawing a line between the nearest points on closed_mask
            for cluster_id in range(5):
                # Build a working mask for just this cluster
                cluster_mask = np.zeros_like(cleaned_rock_mask)

                for idx, contour in enumerate(valid_contours):
                    if labels[idx] == cluster_id:
                        cv2.drawContours(cluster_mask, [contour], -1, 255, thickness=cv2.FILLED)

                # Repeatedly merge the nearest two pieces until only one contour remains
                while True:
                    current_contours, _ = cv2.findContours(
                        cluster_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if len(current_contours) <= 1:
                        break

                    result = nearest_pair(current_contours)
                    if result is None:
                        break

                    p1, p2 = result

                    # Draw the bridge between the two nearest points
                    cv2.line(cluster_mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness=3)

                cleaned_rock_mask = cv2.bitwise_or(cleaned_rock_mask, cluster_mask)
            
            # Use the close mask per contour method to close the mask
            cleaned_rock_mask = close_mask_per_contour(cleaned_rock_mask, alpha=0.005)

            # Save the cleaned rock mask
            out_rock_mask_path = os.path.join(out_rock_subdir, img_name)
            cv2.imwrite(out_rock_mask_path, cleaned_rock_mask)

            board_pixels = int(np.count_nonzero(cleaned_board_mask))
            rock_pixels = int(np.count_nonzero(cleaned_rock_mask))

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

            if not math.isnan(rock_area_square_cm) and not math.isnan(gt_val):
                if rock_area_square_cm > gt_val:
                    direction_err = "overestimate"
                elif rock_area_square_cm < gt_val:
                    direction_err = "underestimate"
                else:
                    direction_err = "exact match"
            else:
                direction_err = float('nan')
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
                'direction_error': direction_err,
                'abs_error': abs_err,
                'pct_error': pct_err,
                'chosen_board_color': board_chosen,
                'chosen_rock_color': rock_chosen,
            })

            if display:
                resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                resized_cleaned_board_mask = cv2.resize(cleaned_board_mask, (0, 0), fx=scale, fy=scale)
                resized_cleaned_rock_mask = cv2.resize(cleaned_rock_mask, (0, 0), fx=scale, fy=scale)
                cv2.imshow('Original', resized_img)
                cv2.imshow('Cleaned Board Mask', resized_cleaned_board_mask)
                cv2.imshow('Cleaned Rock Mask', resized_cleaned_rock_mask)
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
    IMAGE_DIR = r'crayon_test/Crayon Surface Area Photos'
    INIT_BOARD_MASK_DIR = r'Crayon Initial Board Masks'
    INIT_ROCK_MASK_DIR = r'Crayon Initial Rock Masks'
    CLEANED_BOARD_MASK_DIR = r'Crayon Cleaned Board Masks'
    CLEANED_ROCK_MASK_DIR = r'Crayon Cleaned Rock Masks'
    GROUND_TRUTH = r'FoilWeights_2026.xlsx'
    OUT_CSV = r'rock_area_crayon_results.csv'
    DISPLAY = False

    process_folder(IMAGE_DIR, GROUND_TRUTH, OUT_CSV, INIT_BOARD_MASK_DIR, INIT_ROCK_MASK_DIR, CLEANED_BOARD_MASK_DIR, CLEANED_ROCK_MASK_DIR, display=DISPLAY)