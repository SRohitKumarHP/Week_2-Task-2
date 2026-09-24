import cv2
import numpy as np
from scipy.signal import wiener # for Wiener filter
import matplotlib.pyplot as plt


img = cv2.imread('threshoo.jpg') # replace with your file name
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

wiener_img = wiener(gray, (5, 5))
wiener_img = np.uint8(wiener_img)

median_blur = cv2.medianBlur(wiener_img, 3) 

# 4. THRESHOLDING - ADAPTIVE GAUSSIAN
# Handles uneven lighting in old scans
thresh = cv2.adaptiveThreshold(
    median_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 
    15, # block size - must be odd. Bigger = considers larger area
    3 # C - constant subtracted
)

# 5. MORPHOLOGY - CLEAN UP
kernel = np.ones((2, 2), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

cleaned = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(cleaned, -1, kernel_sharp)

cv2.imwrite('1_median.jpg', median_blur)
cv2.imwrite('2_threshold.jpg', thresh)
cv2.imwrite('3_cleaned.jpg', cleaned)
cv2.imwrite('4_final_sharp.jpg', sharpened)

# 8. SHOW COMPARISON
plt.figure(figsize=(12,8))
titles = ['Original', 'Median Blur', 'Adaptive Threshold', 'Final Cleaned']
images = [gray, median_blur, thresh, cleaned]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Done! Check saved files: 1_median.jpg, 2_threshold.jpg, 3_cleaned.jpg, 4_final_sharp.jpg")