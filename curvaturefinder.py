import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = "carved_edges_solo.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform edge detection
edges = cv2.Canny(image, threshold1=150, threshold2=150)

# Display the original image and the detected edges
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Detected Edges')
plt.show()
