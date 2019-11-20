import cv2 as cv
import numpy as np

def sketch(image):
	# Converting the image into grayscale
	img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
	# Removing the blur
	img_gray_blur = cv.GaussianBlur(img_gray, (5, 5), 20)
	# Extracting the edges
	canny_edges = cv.Canny(img_gray_blur, 15, 70)

	ret, mask = cv.threshold(canny_edges, 0, 255, cv.THRESH_BINARY)
	return mask

cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	cv.imwrite("livesketch.jpg", sketch(frame))
	cv.imshow("Live Sketcher", sketch(frame))
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()