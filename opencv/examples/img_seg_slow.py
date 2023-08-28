from __future__ import print_function

import argparse
import random as rng

import cv2 as cv
import numpy as np

rng.seed(12345)


# Load the image
# parser = argparse.ArgumentParser(
#     description="Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
#     Sample code showing how to segment overlapping objects using Laplacian filtering, \
#     in addition to Watershed and Distance Transformation"
# )
# parser.add_argument("--input", help="Path to input image.", default="cards.png")
# args = parser.parse_args()
img_path = "images/1_card_paper_bg_20230817_012700.jpg"
# img_path = "images/many_cards_20230817_012838.jpg"

src = cv.imread(cv.samples.findFile(img_path))
if src is None:
    print("Could not open or find the image:", img_path)
    exit(0)

# Show source image
print("showing img 1")
cv.imshow("Source Image", src)

# Change the background from white to black, since that will help later to extract
# better results during the use of Distance Transform
src[np.all(src == 255, axis=2)] = 0

# Show output image
print("showing img 2")
cv.imshow("Black Background Image", src)

# Create a kernel that we will use to sharpen our image
# an approximation of second derivative, a quite strong kernel
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype("uint8")
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

print("showing img 3")
# cv.imshow('Laplace Filtered Image', imgLaplacian)
cv.imshow("New Sharped Image", imgResult)

# Create binary image from source image
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print("showing img 4")
cv.imshow("Binary Image", bw)

# Perform the distance transform algorithm
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
print("showing img 5")
cv.imshow("Distance Transform Image", dist)

# Threshold to obtain the peaks
# This will be the markers for the foreground objects
_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

# Dilate a bit the dist image
kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
print("showing img 6")
cv.imshow("Peaks", dist)

# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = dist.astype("uint8")

# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)

# Draw the foreground markers
for i in range(len(contours)):
    cv.drawContours(markers, contours, i, (i + 1), -1)

# Draw the background marker
cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
markers_8u = (markers * 10).astype("uint8")
print("showing img 7")
cv.imshow("Markers", markers_8u)

# Perform the watershed algorithm
cv.watershed(imgResult, markers)

# mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype("uint8")
mark = cv.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
# cv.imshow('Markers_v2', mark)

# Generate random colors
colors = []
for contour in contours:
    colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i, j]
        if index > 0 and index <= len(contours):
            dst[i, j, :] = colors[index - 1]

# Visualize the final image
print("showing final img")
cv.imshow("Final Result", dst)

cv.waitKey()
