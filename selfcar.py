import cv2
import numpy as np


def canny(frame):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(gray, 50, 150)
	return canny


def region_of_interest(canny_frame):
	height = canny_frame.shape[0]
	mask = np.zeros_like(canny_frame)

	triangle = np.array([[

		# First arm of triangle
		(200, height),
		# Base
		(550, 250),
		# Third arm of triangle
		(1100, height)
		]])
	cv2.fillPoly(mask, triangle, 255)
	masked_image = cv2.bitwise_and(canny_frame, mask)
	return masked_image


def houghLines(cropped_canny):
	return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
		np.array([]), minLineLength=40, maxLineGap=5)


def display_lines(img, lines):
	line_image = np.zeros_like(img)
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 10)

	return line_image

def make_points(image, line):
	slope, intercept = line
	y1 = int(image.shape[0])
	y2 = int(y1 * 0.5)
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	if lines is None:
		return None

	for line in lines:
		for x1, y1, x2, y2 in line:
			fit = np.polyfit((x1, x2), (y1, y2), 1)
			slope = fit[0]
			intercept = fit[1]

			if slope < 0:
				left_fit.append((slope, intercept))
			else:
				right_fit.append((slope, intercept))

	left_fit_avg = np.average(left_fit, axis=0)
	right_fit_avg = np.average(right_fit, axis=0)

	left_line  = make_points(image, left_fit_avg)
	right_line  = make_points(image, right_fit_avg)

	averaged_lines = [left_line, right_line]
	return averaged_lines

def addWeighted(frame, line_image):
	return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


cap = cv2.VideoCapture('test2.mp4')

while True:
	_, frame = cap.read()
	canny_frame = canny(frame)
	roi = region_of_interest(canny_frame)
	lines = houghLines(roi)
	avg_lines = average_slope_intercept(frame, lines)
	line_image = display_lines(frame, avg_lines)
	combo = addWeighted(frame, line_image)

	cv2.imshow('frame', combo)
	if cv2.waitKey(1) & 0XFF == ord('q'):
		break



cv2.destroyAllWindows()
