import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "carved_edges_solo.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image
_, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the rectangle enclosing the largest contour
largest_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(largest_contour)

# Draw the rectangle
box = cv2.boxPoints(rect)
box = np.int0(box)
image_with_rect = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_rect, [box], 0, (255, 0, 0), 2)

# Display the image with the fitted rectangle
plt.imshow(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB))
plt.show()
