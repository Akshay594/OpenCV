import cv2
import numpy as np

## Color Filtering
## Smoothing & Blurring
## Erosion, Dilation, Opening, Closing
## Perspective Transform



# cap = cv2.VideoCapture(0)

# while True:
# 	_, frame = cap.read()
# 	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 	lower_red = np.array([30, 150, 50])
# 	upper_red = np.array([255, 255, 180])

# 	mask = cv2.inRange(hsv, lower_red, upper_red)
# 	res = cv2.bitwise_and(frame, frame, mask=mask)

# 	# kernel = np.ones((15, 15), np.uint8)/225
# 	# smoothed = cv2.filter2D(res, -1, kernel)

# 	median_blur = cv2.medianBlur(res, 15)
# 	gaussian_blur = cv2.GaussianBlur(res, (15, 15), 0)

# 	kernel = np.ones((3, 3), np.uint8)

# 	erosion = cv2.erode(mask, kernel, iterations=1)
# 	dilation = cv2.dilate(mask, kernel, iterations=1)

# 	# Removes false positive
# 	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# 	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 	# cv2.imshow('erosion', erosion)
# 	# cv2.imshow('dilate', dilation)

# 	# cv2.imshow('smoothed', smoothed)
# 	# cv2.imshow('medianBlur', median_blur)
# 	# cv2.imshow('gaussian_blur', gaussian_blur)
# 	cv2.imshow('res', res)
# 	cv2.imshow('opening', opening)
# 	cv2.imshow('closing', closing)
# 	if cv2.waitKey(1) & 0Xff == ord('q'):
# 		break


# cv2.destroyAllWindows()
# cap.release()



image = cv2.imread('book.jpg')
cv2.circle(image, (222, 250), 1, (0, 0, 255), 3)
cv2.circle(image, (464, 84), 1, (0, 0, 255), 3)
cv2.circle(image, (544, 589), 1, (0, 0, 255), 3)
cv2.circle(image, (788, 391), 1, (0, 0, 255), 3)

pts1 = np.float32([[222, 250], [464, 84], [544, 589], [788, 391]])
pts2 = np.float32([[0, 0], [400, 0], [0, 800], [400, 800]])

M = cv2.getPerspectiveTransform(pts1, pts2)
transformed_image = cv2.warpPerspective(image, M, (400, 800))


cv2.imshow('image', image)
cv2.imshow('transformed_image', transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()