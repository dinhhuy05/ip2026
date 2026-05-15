import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image
# Read original image
img = cv2.imread('grayscale-v1.png')

# Convert BGR to RGB for displaying
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Compute histogram of grayscale image
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(figsize=(6, 4))
plt.plot(hist)
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# Apply histogram equalization
equalized = cv2.equalizeHist(gray)
# Histogram of equalized image
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

plt.figure(figsize=(6, 4))
plt.plot(hist_eq)
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# Resize grayscale image to 150%
scale_percent = 150

width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)

resized = cv2.resize(gray, (width, height))
# Display all required images
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Grayscale image
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Resized grayscale image
plt.subplot(2, 3, 3)
plt.imshow(resized, cmap='gray')
plt.title('Resized Grayscale Image')
plt.axis('off')

# Equalized image
plt.subplot(2, 3, 4)
plt.imshow(equalized, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Side by side comparison
comparison = np.hstack((gray, equalized))

plt.subplot(2, 3, 5)
plt.imshow(comparison, cmap='gray')
plt.title('Gray vs Equalized')
plt.axis('off')

plt.tight_layout()
plt.show()
# Display histograms together
plt.figure(figsize=(12, 5))

# Original histogram
plt.subplot(1, 2, 1)
plt.plot(hist)
plt.title('Original Histogram')

# Equalized histogram
plt.subplot(1, 2, 2)
plt.plot(hist_eq)
plt.title('Equalized Histogram')

plt.show()
# Print image information
# Original image information
print("ORIGINAL IMAGE INFORMATION")
print("Size:", img.shape)
print("Number of channels:", img.shape[2])
print("Data type:", img.dtype)
print("Minimum pixel value:", img.min())
print("Maximum pixel value:", img.max())
# Gray scale information
print("\nGRAYSCALE IMAGE INFORMATION")
print("Size:", gray.shape)
print("Number of channels: 1")
print("Data type:", gray.dtype)
print("Minimum pixel value:", gray.min())
print("Maximum pixel value:", gray.max())
