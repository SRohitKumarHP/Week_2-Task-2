import cv2
import numpy as np

img = cv2.imread('threshoo.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)

th = cv2.adaptiveThreshold(
    blur,                       # input image
    255,                        # value for white pixels
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21,                         # neighborhood size (must be odd)
    10                          # constant subtracted from local mean
)

kernel_open = np.ones((3,3), np.uint8)

opened = cv2.morphologyEx(
    th,
    cv2.MORPH_OPEN,
    kernel_open
)

kernel_close = np.ones((3,3), np.uint8)

clean = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel_close
)

eroded = cv2.erode(clean, kernel_close, iterations=1)
# dilated = cv2.dilate(eroded, kernel_close, iterations=1)
# gb = cv2.medianBlur(dilated, 3)

cv2.imwrite('clean_document.jpg', clean)

cv2.imshow('Original', img)
cv2.imshow('CleanDoc', clean)

print('Document cleaned successfully!')
print('Final output saved as clean_document.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()