import random as rng

import cv2 as cv
import numpy as np

rng.seed(12345)

img_path = "images/1_card_paper_bg_20230817_012700.jpg"
# img_path = "images/many_cards_20230817_012838.jpg"

image = cv.imread(cv.samples.findFile(img_path))

src_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

threshold = 100
canny_output = cv.Canny(src_gray, threshold, threshold * 2)
contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

cv.imshow("Contours", drawing)
# cv.imwrite("../images/output_find_contours.png", drawing)
cv.waitKey()
cv.destroyAllWindows()
