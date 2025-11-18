import cv2

IMAGE = r"/home/deepuser/Documents/OneDrive_1_9-25-2025/Surface Area Photos/15330_WillowBrook_Rocks_062525_iPad.jpg"

# Resize factor
SCALE = 0.2

# Load BGR image
img_bgr = cv2.imread(IMAGE)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE}")

# Convert to HSV
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Create a resized copy for display
display_img = cv2.resize(img_bgr, (0, 0), fx=SCALE, fy=SCALE)

def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Convert coords from display to original image
        orig_x = int(x / SCALE)
        orig_y = int(y / SCALE)

        # Bounds check
        orig_x = min(orig_x, img_bgr.shape[1] - 1)
        orig_y = min(orig_y, img_bgr.shape[0] - 1)

        h, s, v = hsv[orig_y, orig_x]
        print(f"HSV at ({orig_x},{orig_y}): H={h}, S={s}, V={v}")

cv2.namedWindow("HSV Inspector")
cv2.setMouseCallback("HSV Inspector", show_hsv)

while True:
    cv2.imshow("HSV Inspector", display_img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()