from __future__ import division, print_function

import random as rng

import cv2 as cv
import numpy as np

rng.seed(12345)


img_path = "images/1_card_paper_bg_20230817_012700.jpg"
# img_path = "images/many_cards_20230817_012838.jpg"

image = cv.imread(cv.samples.findFile(img_path))

# Convert image to gray and blur it
src_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

threshold = 100
canny_output = cv.Canny(src_gray, threshold, threshold * 2)
contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # cv.RETR_EXTERNAL

# Get the moments
mu = [None] * len(contours)
for i in range(len(contours)):
    mu[i] = cv.moments(contours[i])

# Get the mass centers
mc = [None] * len(contours)
for i in range(len(contours)):
    # add 1e-5 to avoid division by zero
    mc[i] = (mu[i]["m10"] / (mu[i]["m00"] + 1e-5), mu[i]["m01"] / (mu[i]["m00"] + 1e-5))

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours, i, color, 2)
    cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)

# # Calculate the area with the moments 00 and compare with the result of the OpenCV function
# for i in range(len(contours)):
#     print(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f" % (i, mu[i]["m00"], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))

cv.imshow("Contours", drawing)
# cv.imwrite("../images/output_moments.png", drawing)
cv.waitKey()
cv.destroyAllWindows()
