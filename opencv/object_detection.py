import cv2

img_path = "images/1_card_paper_bg_20230817_012700.jpg"
# img_path = "images/many_cards_20230817_012838.jpg"

image = cv2.imread(cv2.samples.findFile(img_path), cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 200)
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # hull = cv2.convexHull(contour)
    # epsilon = 0.05 * cv2.arcLength(contour, True)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detected Cards", image)
# cv2.imwrite("images/output_object_detection.png", image)
cv2.waitKey()
cv2.destroyAllWindows()
