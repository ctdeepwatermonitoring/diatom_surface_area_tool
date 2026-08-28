import cv2
import numpy as np
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster import AgglomerativeClustering

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
        print(len(pts_xy))
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



# Define HSV range for purple color
lower_purple = np.array([110, 0, 20])
upper_purple = np.array([160, 100, 150])

# Define HSV range for red color
lower_red1 = np.array([0, 40, 40])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([165, 40, 70])
upper_red2 = np.array([180, 255, 255])

# Define HSV range for yellow color
lower_yellow = np.array([20, 10, 5])
upper_yellow = np.array([40, 255, 255])

# Define HSV range for blue color
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([112, 255, 255])

# Specify input image path
input_img_path = r"C:/Users/deepuser/Documents/diatom_surface_area_tool/crayon_test/Crayon Surface Area Photos/20636_NonnewaugRiverNNT_Rocks_062526.JPG"

# Load the image
img_bgr = cv2.imread(input_img_path)

# Convert to HSV color space
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Specify which mask to use
mask_color = 'red' # Change to 'purple', 'red', 'blue', or 'yellow' as needed

# Specify rocks or board
# Set to 'rocks' or 'board'
# If 'board', no clustering will be used
rocks_board = 'board'

# Create a mask for the specified color
if mask_color == 'purple':
    color_mask = cv2.inRange(img_hsv, lower_purple, upper_purple)
elif mask_color == 'red':
    red_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    color_mask = cv2.bitwise_or(red_mask1, red_mask2)
elif mask_color == 'yellow':
    color_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
elif mask_color =='blue':
    color_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

#Defining minimum contour area to filter out small contours
img_area = img_bgr.shape[0] * img_bgr.shape[1]
min_contour_area = 0.0002 * img_area

blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
cv2.imshow("Test", cv2.resize(blue_mask, (0,0), fx = 0.2, fy = 0.2))
cv2.waitKey(0)
cv2.destroyAllWindows()
blue_contours, blue_hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for blue_contour in blue_contours:
    if cv2.contourArea(blue_contour) < min_contour_area:
        cv2.drawContours(blue_mask, [blue_contour], -1, 0, thickness=cv2.FILLED)

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41,41))
opened_blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, open_kernel)
closed_blue_mask = cv2.morphologyEx(opened_blue_mask, cv2.MORPH_CLOSE, close_kernel)
closed_blue_mask = close_mask_per_contour(closed_blue_mask, alpha=0.005)
cv2.imshow("Test", cv2.resize(closed_blue_mask, (0,0), fx = 0.2, fy = 0.2))
cv2.waitKey(0)
cv2.destroyAllWindows()
color_mask = cv2.bitwise_and(closed_blue_mask, color_mask)

#Find contours in the closed mask
contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Bit flip white parts of the mask that are below min_contour_area
for contour in contours:
    if cv2.contourArea(contour) < min_contour_area:
        cv2.drawContours(color_mask, [contour], -1, 0, thickness=cv2.FILLED)


#Opening the image to remove noise
opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel)

#Closing the image to connect the mask and fill in holes
closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)

if rocks_board == 'rocks':
    #Getting closed contours
    closed_contours, closed_hierarchy = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Getting the centroids of the closed contours
    contour_centroids = []
    for contour in closed_contours:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        contour_centroids.append((cx, cy))

    #Drawing centroids on the original image
    for centroid in contour_centroids:
        cv2.circle(img_bgr, centroid, 5, (0, 255, 0), -1)

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
        cluster_mask = np.zeros_like(closed_mask)

        for idx, contour in enumerate(closed_contours):
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

        closed_mask = cv2.bitwise_or(closed_mask, cluster_mask)


#Draw the centroids on the original image with different colors based on their cluster labels
'''colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
for i, centroid in enumerate(contour_centroids):
    color = colors[labels[i]]
    cv2.circle(img_bgr, centroid, 5, color, -1)'''

#Use close_mask_per_contour method to close the mask
closed_mask = close_mask_per_contour(closed_mask, alpha=0.005)

#Scale to resize the images before displaying
scale = 0.2
display_img = cv2.resize(img_bgr, (0,0), fx = scale, fy = scale)
display_mask = cv2.resize(color_mask, (0,0), fx = scale, fy = scale)
display_closed_mask = cv2.resize(closed_mask, (0,0), fx = scale, fy = scale)

# Display the resized mask alongside the resized original image
cv2.imshow("Original Image", display_img)
cv2.imshow("Color Mask", display_mask)
cv2.imshow("Closed Color Mask", display_closed_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()