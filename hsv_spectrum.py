import numpy as np
import cv2

spectrum_hsv = np.zeros((179, 179, 3), dtype = np.uint8)

for col in range(179):
    spectrum_hsv[:, col, 0] = col
    spectrum_hsv[:, col, 1] = 255
    spectrum_hsv[:, col, 2] = 255

spectrum_bgr = cv2.cvtColor(spectrum_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Hue Spectrum", spectrum_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()