from __future__ import division, print_function

import argparse
import random as rng

import cv2 as cv
import numpy as np

rng.seed(12345)


def wait_for_key_press(win_name, image):
    print("waiting", end=" ")
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("key pressed")


def thresh_callback(val):
    threshold = val

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
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

    # Draw contours - [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    #cheez
    print(len(contours))
    # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    # tmp = len(contours) - 1
    # cv.drawContours(drawing, contours, tmp, color, 2)
    # cv.circle(drawing, (int(mc[tmp][0]), int(mc[tmp][1])), 4, color, -1)
    # wait_for_key_press("Contours", drawing)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)

        # 1643 total
        # Contour[1642] - Area (M_00) = 119.00 - Area OpenCV: 119.00 - Length: 12485.61

        # 6667 total
        if i > 6660:
            print(f"i: {i}")
            print("\n\nCHEEEEEEEEEEZZZZ\n\n")
            wait_for_key_press("Contours", drawing)
        # elif i > 1600:
        #     print(f"i: {i}")
        #     wait_for_key_press("Contours", drawing)
        elif i % 500 == 0:
            print(f"i: {i}")
            wait_for_key_press("Contours", drawing)

    # Show in a window
    cv.imshow("Contours", drawing)

    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        print(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f" % (i, mu[i]["m00"], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))


# class Args: input = "images/1_card_paper_bg_20230817_012700.jpg"
class Args: input = "images/many_cards_20230817_012838.jpg"
args = Args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print("Could not open or find the image:", args.input)
    exit(0)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

source_window = "Source"
cv.namedWindow(source_window)
cv.imshow(source_window, src)

# trackbar
max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar("Canny Thresh:", source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
